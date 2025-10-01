import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

# Logger module for the project. Writes logs under artifacts/logs and to console.
# Provides a backward-compatible log(message: str) function that logs at INFO level.

_MODULE_LOGGER_NAME = "tf3"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_BASE_DIR, "artifacts", "logs")
_DEFAULT_LOG_FILE = os.path.join(_LOG_DIR, "tf3.log")

_configured: bool = False


def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def _create_file_handler(log_file_path: str) -> logging.Handler:
    handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _create_stream_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def configure_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure the project logger.

    - Writes to artifacts/logs/tf3.log (rotated nightly, 14 backups) by default
    - Also logs to console (stream handler) at the provided level
    - Idempotent: safe to call multiple times
    """
    global _configured
    if _configured:
        return

    _ensure_log_dir()

    effective_log_file = log_file or _DEFAULT_LOG_FILE

    print(f"Logging to {effective_log_file}")

    root_logger = logging.getLogger(_MODULE_LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(_create_file_handler(effective_log_file))

    if not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, TimedRotatingFileHandler)
        for h in root_logger.handlers
    ):
        root_logger.addHandler(_create_stream_handler(level))

    root_logger.propagate = False
    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a named logger under the project root logger.

    Examples:
        get_logger() -> "tf3"
        get_logger("training") -> "tf3.training"
    """
    if not _configured:
        configure_logger(log_file=f"{_LOG_DIR}/{name}.log")
    logger_name = _MODULE_LOGGER_NAME if not name else f"{_MODULE_LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


# Backward-compatible simple function


def log(message: str) -> None:
    """Log a message at INFO level (backward compatible)."""
    get_logger().info(message)


# Convenience wrappers


def debug(message: str) -> None:
    get_logger().debug(message)


def info(message: str) -> None:
    get_logger().info(message)


def warning(message: str) -> None:
    get_logger().warning(message)


def error(message: str) -> None:
    get_logger().error(message)


def exception(message: str) -> None:
    """Log an exception with traceback. Call inside an except block."""
    get_logger().exception(message)
