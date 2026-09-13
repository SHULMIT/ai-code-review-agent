"""Process startup entry point for the Code Review Agent."""

import asyncio
import logging
import sys

import uvicorn

from review_agent.app import create_app
from review_agent.config import Settings
from review_agent.logging import configure_logging, log_event

app = create_app()


def run() -> None:
    """Load runtime settings and start the Uvicorn process.

    On Windows, uvicorn.run() always selects asyncio's ProactorEventLoop
    (see uvicorn/loops/asyncio.py) unless running with --reload or multiple
    workers, neither of which this service uses - and psycopg's async
    driver refuses to run under ProactorEventLoop at all, raising
    psycopg.InterfaceError the first time it opens a database connection.
    Constructing the SelectorEventLoop explicitly and driving the server
    through it directly (uvicorn.Server.serve(), not uvicorn.run()) avoids
    that hardcoded selection entirely. Every other platform still reaches
    the same uvicorn.Server.serve() call, just through plain asyncio.run()
    instead of an explicit Windows-only event loop policy.
    """

    settings = Settings()
    settings.validate_runtime_configuration()
    service_logger = configure_logging(settings.log_level)
    log_event(
        service_logger,
        logging.INFO,
        "service_starting",
    )
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    run()
