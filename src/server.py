#!/usr/bin/env python3
"""TranslateGemma 27B IT REST server for Kaggle TPU v5e-8.

The coordinator process runs HTTP/queue/lifecycle code only. JAX, Keras and
KerasHub are imported by the single spawned TPU worker after TPU setup.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal

from waitress import serve

from translategemma_server.api.app import Runtime, create_app
from translategemma_server.core.config import Config
from translategemma_server.core.paths import LOG_DIR, configure_logging
from translategemma_server.workers.manager import TranslationManager

logger = logging.getLogger("translategemma_server")


class CoordinatorShutdown(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"shutdown signal {signum}")


def _shutdown_signal(signum, _frame) -> None:
    raise CoordinatorShutdown(signum)


def _install_shutdown_signals() -> dict[int, object]:
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _shutdown_signal)
    return previous


def _restore_shutdown_signals(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main() -> int:
    mp.freeze_support()
    configure_logging("api", LOG_DIR / "server.log")
    try:
        config = Config.from_env()
    except Exception:
        logger.exception("Invalid configuration")
        return 2

    manager = TranslationManager(config)
    runtime = Runtime(config, manager)
    app = create_app(runtime)
    previous_signals = _install_shutdown_signals()
    try:
        manager.start_async()
        serve(
            app,
            host=config.host,
            port=config.port,
            threads=max(4, config.max_queue_size + 3),
            channel_timeout=max(120, int(config.request_timeout) + 60),
        )
    except CoordinatorShutdown as exc:
        logger.info("Coordinator received shutdown signal %s", exc.signum)
    finally:
        _restore_shutdown_signals(previous_signals)
        manager.shutdown(False, config.shutdown_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
