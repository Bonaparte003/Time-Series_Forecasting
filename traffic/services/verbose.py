"""Shared verbose logging for management commands and services."""

from __future__ import annotations

from collections.abc import Callable

LogFn = Callable[[str], None] | None

VERBOSE_HELP = (
    "Print step-by-step progress. "
    "Also enabled when Django verbosity is 2+ (e.g. manage.py <command> -v 2)."
)


def is_verbose(options: dict) -> bool:
    return bool(options.get("verbose")) or options.get("verbosity", 1) >= 2


def command_log(
    options: dict,
    write: Callable[[str], None],
    *,
    default: bool = False,
) -> LogFn:
    """
    Return a log function (stdout.write) or None.

    ``--quiet`` disables logging. Otherwise logging is on when ``--verbose``,
    Django ``-v 2+``, or ``default=True`` (experiments / forecast).
    """
    if options.get("quiet"):
        return None
    if is_verbose(options) or default:
        return write
    return None


def add_verbose_argument(parser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=VERBOSE_HELP,
    )


def add_quiet_argument(parser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step progress output.",
    )
