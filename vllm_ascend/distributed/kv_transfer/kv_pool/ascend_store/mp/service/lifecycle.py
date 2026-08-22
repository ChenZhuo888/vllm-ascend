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


class ServiceLifecycleManager(Generic[IdentityT, ServiceT]):
    """Manage the registration, lease, expiration, and closure of one service type.

    Internal orchestration uses ``find`` without renewing a lease. Requests from
    a service owner use ``get_for_session``. Background expiration and manager
    shutdown use ``owner_close_handler`` when the service has an owner lane.
    """

    def __init__(
        self,
        service_name: str,
        close_service: Callable[[ServiceT], None],
        lease_timeout_s: float,
        check_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str | None = None,
        owner_close_handler: Callable[[IdentityT, ServiceT], None] | None = None,
    ):
        if not service_name:
            raise ValueError("service_name must not be empty")
        if lease_timeout_s <= 0:
            raise ValueError(f"lease_timeout_s must be greater than 0, got {lease_timeout_s}")
        if check_interval_s <= 0:
            raise ValueError(f"check_interval_s must be greater than 0, got {check_interval_s}")

        self._service_name = service_name
        self._close_service = close_service
        self._lease_timeout_s = lease_timeout_s
        self._check_interval_s = check_interval_s
        self._clock = clock
        self._thread_name = thread_name or f"{service_name.lower()}-service-lifecycle"
        self._owner_close_handler = owner_close_handler

        self._lock = threading.RLock()
        self._services: dict[IdentityT, _ServiceEntry[ServiceT]] = {}
        self._registering: dict[IdentityT, _RegistrationFlight[ServiceT]] = {}
        self._expiring: dict[IdentityT, _ServiceEntry[ServiceT]] = {}
        self._recoverable_sessions: dict[IdentityT, _RecoverableSession] = {}
        self._retired_sessions: dict[IdentityT, set[str]] = {}
        self._closed = False

        self._expiration_lock = threading.RLock()
        self._maintenance_lock = threading.Lock()
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._services)

    @property
    def is_running(self) -> bool:
        with self._maintenance_lock:
            return self._maintenance_thread is not None and self._maintenance_thread.is_alive()

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
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            if identity in self._expiring:
                raise ServiceBusyError(f"{self._service_name} {identity!r} is being expired")

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

    def renew(self, identity: IdentityT, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            entry = self._services.get(identity)
            if entry is None:
                return False
            self._validate_session(identity, session_id, entry.session_id)
            entry.last_seen = self._clock()
            return True

    def find(self, identity: IdentityT) -> ServiceT | None:
        """Return a service without validating or renewing its owner session."""
        with self._lock:
            self._raise_if_closed()
            entry = self._services.get(identity)
            return None if entry is None else entry.service

    def get_for_session(self, identity: IdentityT, session_id: str) -> ServiceT | None:
        """Validate the owner session, renew its lease, and return the service."""
        self._validate_session_id(session_id)
        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            entry = self._services.get(identity)
            if entry is None:
                return None
            self._validate_session(identity, session_id, entry.session_id)
            entry.last_seen = self._clock()
            return entry.service

    def unregister(self, identity: IdentityT, session_id: str) -> bool:
        self._validate_session_id(session_id)
        service = None

        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)

            expiring = self._expiring.get(identity)
            if expiring is not None:
                self._validate_session(identity, session_id, expiring.session_id)
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

    def expire_leases(self) -> int:
        with self._expiration_lock:
            stale_before = self._clock() - self._lease_timeout_s
            with self._lock:
                if self._closed:
                    return 0
                expired_services = [
                    (identity, entry) for identity, entry in self._services.items() if entry.last_seen <= stale_before
                ]
                for identity, entry in expired_services:
                    del self._services[identity]
                    self._expiring[identity] = entry

            for identity, entry in expired_services:
                self._close_on_owner_safely(identity, entry.service)
                self._finish_expiration(identity, entry)
            return len(expired_services)

    def start_maintenance(self) -> None:
        with self._lock:
            self._raise_if_closed()
            with self._maintenance_lock:
                if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
                    return

                self._maintenance_stop.clear()
                self._maintenance_thread = threading.Thread(
                    target=self._maintenance_loop, daemon=True, name=self._thread_name
                )
                self._maintenance_thread.start()

    def stop_maintenance(self, wait: bool = True) -> None:
        with self._maintenance_lock:
            thread = self._maintenance_thread
            if thread is None:
                return
            self._maintenance_stop.set()

        if not wait:
            return
        if thread is not threading.current_thread():
            thread.join()

        with self._maintenance_lock:
            if self._maintenance_thread is thread:
                self._maintenance_thread = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self.stop_maintenance()
        with self._expiration_lock, self._lock:
            services = [(identity, entry.service) for identity, entry in self._services.items()]
            services.extend((identity, entry.service) for identity, entry in self._expiring.items())
            self._services.clear()
            self._registering.clear()
            self._expiring.clear()
            self._recoverable_sessions.clear()
            self._retired_sessions.clear()

        for identity, service in services:
            self._close_on_owner_safely(identity, service)

    def _maintenance_loop(self) -> None:
        while not self._maintenance_stop.wait(self._check_interval_s):
            try:
                self.expire_leases()
            except Exception:
                logger.exception("%s service lifecycle maintenance failed", self._service_name)

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
        self._raise_if_closed()
        current_flight = self._registering.get(identity)
        if current_flight is not flight:
            raise RuntimeError(f"{self._service_name} {identity!r} registration is no longer active")

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

    def _finish_expiration(self, identity: IdentityT, entry: _ServiceEntry[ServiceT]) -> None:
        with self._lock:
            if self._expiring.get(identity) is not entry:
                return
            del self._expiring[identity]
            if entry.session_id not in self._retired_sessions.get(identity, ()):
                self._recoverable_sessions[identity] = _RecoverableSession(entry.session_id, entry.fingerprint)

    def _retire_session_locked(self, identity: IdentityT, session_id: str) -> None:
        self._retired_sessions.setdefault(identity, set()).add(session_id)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self._service_name} lifecycle manager is closed")

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

    def _close_on_owner_safely(self, identity: IdentityT, service: ServiceT) -> None:
        if self._owner_close_handler is None:
            self._close_service_safely(service)
            return

        try:
            self._owner_close_handler(identity, service)
        except Exception:
            logger.exception("Failed to close %s service %r on its owner", self._service_name, service)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str):
            raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
        if not session_id:
            raise ValueError("session_id must not be empty")
