"""Wire protocol definitions for AscendStore multiprocess communication."""

import enum
from collections.abc import Iterable, Sequence

from .error import MPProtocolError

MultipartMessage = tuple[bytes, ...]


class SystemMethod(str, enum.Enum):
    PING = "PING"
    ECHO = "ECHO"


class ResponseStatus(str, enum.Enum):
    OK = "OK"
    ERROR = "ERROR"
    BUSY = "BUSY"


def normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}")

    method_name = method.value if isinstance(method, enum.Enum) else method
    if not method_name:
        raise ValueError("method must not be empty")
    return method_name


def encode_method(method: str) -> bytes:
    return normalize_method(method).encode()


def decode_method(data: bytes) -> str:
    _validate_frame(data, "method")

    try:
        method = data.decode()
    except UnicodeDecodeError as exc:
        raise MPProtocolError("Method frame is not valid UTF-8") from exc

    if not method:
        raise MPProtocolError("Method frame must not be empty")
    return method


def encode_response_status(status: ResponseStatus) -> bytes:
    return status.value.encode()


def decode_response_status(data: bytes) -> ResponseStatus:
    _validate_frame(data, "response status")

    try:
        return ResponseStatus(data.decode())
    except (UnicodeDecodeError, ValueError) as exc:
        raise MPProtocolError(f"Invalid response status: {data!r}") from exc


def encode_request(request_id: bytes, method: str, payloads: Iterable[bytes] = ()) -> MultipartMessage:
    _validate_frame(request_id, "request ID")
    return request_id, encode_method(method), *_normalize_payloads(payloads)


def decode_request(frames: Sequence[bytes]) -> tuple[bytes, str, tuple[bytes, ...]]:
    if len(frames) < 2:
        raise MPProtocolError(f"Expected [request_id, method, *payloads], got {len(frames)} frames")

    request_id, method_frame, *payloads = frames
    _validate_frame(request_id, "request ID")
    return request_id, decode_method(method_frame), _normalize_payloads(payloads)


def encode_response(
    request_id: bytes,
    method: str,
    status: ResponseStatus,
    payloads: Iterable[bytes] = (),
) -> MultipartMessage:
    _validate_frame(request_id, "request ID")
    return request_id, encode_method(method), encode_response_status(status), *_normalize_payloads(payloads)


def decode_response(
    frames: Sequence[bytes],
) -> tuple[bytes, str, ResponseStatus, tuple[bytes, ...]]:
    if len(frames) < 3:
        raise MPProtocolError(f"Expected [request_id, method, status, *payloads], got {len(frames)} frames")

    request_id, method_frame, status_frame, *payloads = frames
    _validate_frame(request_id, "request ID")
    return (
        request_id,
        decode_method(method_frame),
        decode_response_status(status_frame),
        _normalize_payloads(payloads),
    )


def _normalize_payloads(payloads: Iterable[bytes]) -> tuple[bytes, ...]:
    try:
        normalized = tuple(payloads)
    except TypeError as exc:
        raise MPProtocolError("Payloads must be an iterable of bytes") from exc

    for index, payload in enumerate(normalized):
        _validate_frame(payload, f"payload {index}")
    return normalized


def _validate_frame(frame: bytes, name: str) -> None:
    if not isinstance(frame, bytes):
        raise MPProtocolError(f"{name} must be bytes, got {type(frame).__name__}")
