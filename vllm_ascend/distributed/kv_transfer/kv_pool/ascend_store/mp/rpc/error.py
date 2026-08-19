class MPError(Exception):
    pass


class MPProtocolError(MPError):
    pass


class MPRemoteError(MPError):
    pass


class MPClientClosedError(MPError):
    pass


class MPServerUnavailableError(MPError, ConnectionError):
    pass


class MPRequestTimeoutError(MPError, TimeoutError):
    pass


class MPServerBusyError(MPError):
    pass
