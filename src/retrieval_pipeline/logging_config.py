"""Logging configuration for the retrieval pipeline.

Call :func:`setup_logging` once at application startup. All modules that do
``from loguru import logger`` automatically inherit both sinks (console +
rotating file) without any further changes.

Loguru log levels used in this codebase (lowest → highest severity):

    logger.debug(...)    — developer-facing state: counts, sizes, paths
    logger.info(...)     — normal operational milestones
    logger.success(...)  — step completed successfully (green in console)
    logger.warning(...)  — recoverable issue; pipeline continues
    logger.error(...)    — non-fatal failure; action was skipped or degraded
    logger.critical(...) — unrecoverable; process is about to exit

Note: loguru does NOT have .warn() or .fatal() methods.
Use .warning() and .critical() respectively.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Guard flag so calling setup_logging() more than once is safe.
_logging_initialised: bool = False


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
    colorize_console: bool = True,
) -> None:
    """Configure loguru with a console sink and a rotating file sink.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    log_dir : str
        Directory where log files are stored (created automatically).
    log_file : str
        Base filename for the log file inside *log_dir*.
    console_level : str
        Minimum severity written to stderr (default ``"INFO"`` to avoid
        flooding the terminal with debug lines).
    file_level : str
        Minimum severity written to the rotating log file (default
        ``"DEBUG"`` so all detail is captured on disk).
    rotation : str
        When to rotate the log file (size string, time string, or callable).
    retention : str
        How long rotated files are kept before deletion.
    compression : str
        Archive format applied to rotated files (``"zip"``, ``"gz"`` etc.).
    colorize_console : bool
        Whether to emit ANSI colour codes on stderr.
    """
    global _logging_initialised  # noqa: PLW0603
    if _logging_initialised:
        return

    # Remove the default loguru handler so we control format & sinks fully.
    logger.remove()

    # ── Console sink ─────────────────────────────────────────────────────
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_fmt,
        level=console_level,
        colorize=colorize_console,
        enqueue=True,
    )

    # ── Rotating file sink ───────────────────────────────────────────────
    log_path = Path(log_dir) / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | " "{level: <8} | " "{name}:{function}:{line} - " "{message}"
    )
    logger.add(
        str(log_path),
        format=file_fmt,
        level=file_level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        encoding="utf-8",
    )

    _logging_initialised = True
    logger.info(
        "Logging initialised | console>={} | file>={} | sink={}",
        console_level,
        file_level,
        log_path.resolve(),
    )
