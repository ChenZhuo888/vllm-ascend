import logging
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Generic, TypeVar

from .error import RegistrationConflictError, ServiceBusyError, StaleSessionError

logger = logging.getLogger(__name__)

IdentityT = TypeVar("IdentityT", bound=Hashable)
ServiceT = TypeVar("ServiceT")


@dataclass
class _ServiceEntry(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    service: ServiceT
    last_seen: float


@dataclass(frozen=True)
class _RecoverableSession:
    session_id: str
    fingerprint: bytes


@dataclass(frozen=True)
class _RegistrationFlight(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    future: Future[ServiceT]


class ServiceRegistry(Generic[IdentityT, ServiceT]):
    """Manage session-aware service instances for one service type."""

    def __init__(
        self,
        service_name: str,
        close_service: Callable[[ServiceT], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        if not service_name:
            raise ValueError("service_name must not be empty")

        self._service_name = service_name
        self._close_service = close_service
        self._clock = clock
        self._lock = threading.RLock()
        self._services: dict[IdentityT, _ServiceEntry[ServiceT]] = {}
        self._registering: dict[IdentityT, _RegistrationFlight[ServiceT]] = {}
        self._reaping: dict[IdentityT, _ServiceEntry[ServiceT]] = {}
        self._recoverable_sessions: dict[IdentityT, _RecoverableSession] = {}
        self._retired_sessions: dict[IdentityT, set[str]] = {}

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._services)

    def items(self) -> tuple[tuple[IdentityT, ServiceT], ...]:
        with self._lock:
            return tuple((identity, entry.service) for identity, entry in self._services.items())

    def register(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        factory: Callable[[], ServiceT],
    ) -> ServiceT:
        self._validate_session_id(session_id)
        old_service = None

        with self._lock:
            self._raise_if_retired(identity, session_id)
            if identity in self._reaping:
                raise ServiceBusyError(f"{self._service_name} {identity!r} is being reaped")

            entry = self._services.get(identity)
            if entry is not None:
                if entry.session_id == session_id:
                    self._validate_fingerprint(identity, entry.fingerprint, fingerprint)
                    entry.last_seen = self._clock()
                    return entry.service

                self._retire_session_locked(identity, entry.session_id)
                del self._services[identity]
                old_service = entry.service

            flight = self._registering.get(identity)
            if flight is not None:
                if flight.session_id != session_id:
                    raise RegistrationConflictError(
                        f"{self._service_name} {identity!r} is already registering session {flight.session_id!r}"
                    )
                self._validate_fingerprint(identity, flight.fingerprint, fingerprint)
                wait_future = flight.future
            else:
                self._prepare_recoverable_locked(identity, session_id, fingerprint)
                flight = _RegistrationFlight(session_id, fingerprint, Future())
                self._registering[identity] = flight
                wait_future = None

        if wait_future is not None:
            return wait_future.result()

        return self._create_and_publish(identity, session_id, fingerprint, factory, flight, old_service)

    def unregister(self, identity: IdentityT, session_id: str) -> bool:
        self._validate_session_id(session_id)
        service = None

        with self._lock:
            self._raise_if_retired(identity, session_id)

            reaping = self._reaping.get(identity)
            if reaping is not None:
                self._validate_session(identity, session_id, reaping.session_id)
                self._retire_session_locked(identity, session_id)
                return True

            recoverable = self._recoverable_sessions.get(identity)
            if recoverable is not None:
                self._validate_session(identity, session_id, recoverable.session_id)
                del self._recoverable_sessions[identity]
                self._retire_session_locked(identity, session_id)
                return True

            entry = self._services.get(identity)
            if entry is None:
                return False
            self._validate_session(identity, session_id, entry.session_id)

            del self._services[identity]
            self._retire_session_locked(identity, session_id)
            service = entry.service

        self._close_service(service)
        return True

    def touch(self, identity: IdentityT, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            self._raise_if_retired(identity, session_id)
            entry = self._services.get(identity)
            if entry is None:
                return False
            self._validate_session(identity, session_id, entry.session_id)
            entry.last_seen = self._clock()
            return True

    def get(self, identity: IdentityT, session_id: str | None = None) -> ServiceT | None:
        with self._lock:
            if session_id is not None:
                self._validate_session_id(session_id)
                self._raise_if_retired(identity, session_id)

            entry = self._services.get(identity)
            if entry is None:
                return None
            if session_id is not None:
                self._validate_session(identity, session_id, entry.session_id)
                entry.last_seen = self._clock()
            return entry.service

    def reap_stale(self, stale_before: float) -> int:
        with self._lock:
            stale_services = [
                (identity, entry) for identity, entry in self._services.items() if entry.last_seen < stale_before
            ]
            for identity, entry in stale_services:
                del self._services[identity]
                self._reaping[identity] = entry

        for identity, entry in stale_services:
            self._close_service_safely(entry.service)
            self._finish_reap(identity, entry)
        return len(stale_services)

    def close(self) -> None:
        with self._lock:
            services = [entry.service for entry in self._services.values()]
            services.extend(entry.service for entry in self._reaping.values())
            self._services.clear()
            self._registering.clear()
            self._reaping.clear()
            self._recoverable_sessions.clear()
            self._retired_sessions.clear()

        for service in services:
            self._close_service_safely(service)

    def _prepare_recoverable_locked(self, identity: IdentityT, session_id: str, fingerprint: bytes) -> None:
        recoverable = self._recoverable_sessions.get(identity)
        if recoverable is None:
            return
        if recoverable.session_id == session_id:
            self._validate_fingerprint(identity, recoverable.fingerprint, fingerprint)
            return

        self._retire_session_locked(identity, recoverable.session_id)
        del self._recoverable_sessions[identity]

    def _create_and_publish(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        factory: Callable[[], ServiceT],
        flight: _RegistrationFlight[ServiceT],
        old_service: ServiceT | None,
    ) -> ServiceT:
        service = None
        try:
            if old_service is not None:
                self._close_service(old_service)

            service = factory()
            with self._lock:
                self._publish_locked(identity, session_id, fingerprint, flight, service)
        except BaseException as exc:
            self._fail_registration(identity, flight, service, exc)
            raise

        flight.future.set_result(service)
        return service

    def _publish_locked(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        flight: _RegistrationFlight[ServiceT],
        service: ServiceT,
    ) -> None:
        current_flight = self._registering.get(identity)
        assert current_flight is flight

        self._services[identity] = _ServiceEntry(session_id, fingerprint, service, self._clock())
        recoverable = self._recoverable_sessions.get(identity)
        if recoverable is not None and recoverable.session_id == session_id:
            del self._recoverable_sessions[identity]
        del self._registering[identity]

    def _fail_registration(
        self,
        identity: IdentityT,
        flight: _RegistrationFlight[ServiceT],
        service: ServiceT | None,
        exc: BaseException,
    ) -> None:
        with self._lock:
            if self._registering.get(identity) is flight:
                del self._registering[identity]

        if service is not None:
            self._close_service_safely(service)
        if not flight.future.done():
            flight.future.set_exception(exc)

    def _finish_reap(self, identity: IdentityT, entry: _ServiceEntry[ServiceT]) -> None:
        with self._lock:
            if self._reaping.get(identity) is not entry:
                return
            del self._reaping[identity]
            if entry.session_id not in self._retired_sessions.get(identity, ()):
                self._recoverable_sessions[identity] = _RecoverableSession(entry.session_id, entry.fingerprint)

    def _retire_session_locked(self, identity: IdentityT, session_id: str) -> None:
        self._retired_sessions.setdefault(identity, set()).add(session_id)

    def _raise_if_retired(self, identity: IdentityT, session_id: str) -> None:
        if session_id in self._retired_sessions.get(identity, ()):
            raise StaleSessionError(f"{self._service_name} {identity!r} session {session_id!r} has been retired")

    def _validate_session(self, identity: IdentityT, incoming: str, current: str) -> None:
        if incoming != current:
            raise StaleSessionError(
                f"{self._service_name} {identity!r} session {incoming!r} is stale; current session is {current!r}"
            )

    def _validate_fingerprint(self, identity: IdentityT, existing: bytes, incoming: bytes) -> None:
        if existing != incoming:
            raise RegistrationConflictError(
                f"{self._service_name} {identity!r} is already registered with different configuration"
            )

    def _close_service_safely(self, service: ServiceT) -> None:
        try:
            self._close_service(service)
        except Exception:
            logger.exception("Failed to close %s service %r", self._service_name, service)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str):
            raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
        if not session_id:
            raise ValueError("session_id must not be empty")
