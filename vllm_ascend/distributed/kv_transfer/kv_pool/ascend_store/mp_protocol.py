import enum


class RequestType(enum.Enum):
    PING = "PING"
    SHUTDOWN = "SHUTDOWN"
    ECHO = "ECHO"


class ResponseStatus(enum.Enum):
    OK = "OK"
    ERROR = "ERROR"


def encode_request_type(request_type: RequestType) -> bytes:
    return request_type.value.encode()


def decode_request_type(data: bytes) -> RequestType:
    return RequestType(data.decode())


def encode_response_status(status: ResponseStatus) -> bytes:
    return status.value.encode()


def decode_response_status(data: bytes) -> ResponseStatus:
    return ResponseStatus(data.decode())