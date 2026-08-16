#!/usr/bin/env python3
"""TranslateGemma 27B IT REST server for Kaggle TPU v5e-8.

The coordinator process runs HTTP/queue/lifecycle code only. JAX, Keras and
KerasHub are imported by the single spawned TPU worker after TPU setup.
"""
from __future__ import annotations

import logging
import multiprocessing as mp

from waitress import serve

from translategemma_server.api.app import Runtime, create_app
from translategemma_server.core.config import Config
from translategemma_server.core.paths import LOG_DIR, configure_logging
from translategemma_server.workers.manager import TranslationManager

logger = logging.getLogger("translategemma_server")


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
    manager.start_async()
    try:
        serve(
            app,
            host=config.host,
            port=config.port,
            threads=max(4, config.max_queue_size + 3),
            channel_timeout=max(120, int(config.request_timeout) + 60),
        )
    finally:
        manager.shutdown(False, 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
