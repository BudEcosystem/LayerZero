"""
Subprocess worker for isolated backend execution.

This module is invoked as:
    python -m layerzero.isolation.worker <backend_id>

Communication protocol:
- Reads JSON requests from stdin (line-delimited)
- Writes JSON responses to stdout (line-delimited)
- Logs to stderr

Request format:
    {"operation": "ping"}
    {"operation": "execute", "payload": {...}}
    {"operation": "shutdown"}

Response format:
    {"status": "ok", "result": ...}
    {"status": "error", "error": "..."}

This worker is spawned by SubprocessBackend and handles IPC
for isolated model execution to prevent ABI conflicts.
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for shutdown
_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown signal received (signal=%d)", signum)


def process_request(request: dict[str, Any], backend_id: str) -> dict[str, Any]:
    """Process a request and return response.

    Args:
        request: Request dictionary with 'operation' and optional 'payload' keys.
        backend_id: Backend identifier for context.

    Returns:
        Response dictionary with 'status' and 'result' or 'error' keys.
    """
    try:
        operation = request.get("operation")
        payload = request.get("payload", {})

        if operation == "ping":
            return {
                "status": "ok",
                "result": {
                    "pong": True,
                    "backend_id": backend_id,
                },
            }

        elif operation == "execute":
            # Execute backend operation with provided payload
            # This is a placeholder - actual implementations would
            # load and run specific models based on the payload
            result = _execute_backend_operation(payload, backend_id)
            return {"status": "ok", "result": result}

        elif operation == "shutdown":
            global _shutdown_requested
            _shutdown_requested = True
            return {
                "status": "ok",
                "result": {"shutdown": True, "backend_id": backend_id},
            }

        elif operation == "health":
            # Health check endpoint
            return {
                "status": "ok",
                "result": {
                    "healthy": True,
                    "backend_id": backend_id,
                },
            }

        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }

    except Exception as e:
        logger.exception("Error processing request")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _execute_backend_operation(
    payload: dict[str, Any], backend_id: str
) -> dict[str, Any]:
    """Execute backend operation.

    This is a placeholder implementation. Concrete backends would override
    this to provide actual model inference or processing logic.

    Args:
        payload: Operation payload with configuration and data.
        backend_id: Backend identifier.

    Returns:
        Result dictionary.
    """
    # Extract common fields
    operation_type = payload.get("type", "unknown")
    data = payload.get("data")

    logger.debug(
        "Executing operation type=%s for backend=%s",
        operation_type,
        backend_id,
    )

    # Placeholder response - real implementation would process data
    return {
        "operation_type": operation_type,
        "backend_id": backend_id,
        "processed": True,
        "data": data,
    }


def main(backend_id: str) -> int:
    """Main worker loop.

    Args:
        backend_id: Backend identifier for logging.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Configure logging to stderr
    logging.basicConfig(
        level=logging.INFO,
        format=f"[worker:{backend_id}] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Worker started for backend: %s", backend_id)
    logger.info("Ready to receive requests on stdin")

    exit_code = 0

    try:
        for line in sys.stdin:
            # Check for shutdown request
            if _shutdown_requested:
                logger.info("Shutdown requested, exiting main loop")
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                logger.debug("Received request: %s", request.get("operation", "unknown"))

                response = process_request(request, backend_id)

                # Write response to stdout (line-delimited JSON)
                response_json = json.dumps(response)
                print(response_json, flush=True)

                # Check if this was a shutdown request
                if _shutdown_requested:
                    logger.info("Shutdown completed")
                    break

            except json.JSONDecodeError as e:
                error_response = {
                    "status": "error",
                    "error": f"Invalid JSON: {e}",
                }
                print(json.dumps(error_response), flush=True)
                logger.warning("Invalid JSON received: %s", e)

    except KeyboardInterrupt:
        logger.info("Worker interrupted by keyboard")
    except BrokenPipeError:
        # Parent process closed the pipe - normal shutdown
        logger.info("Parent process closed pipe")
    except Exception as e:
        logger.exception("Worker crashed: %s", e)
        exit_code = 1

    logger.info("Worker exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m layerzero.isolation.worker <backend_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    backend_id = sys.argv[1]
    exit_code = main(backend_id)
    sys.exit(exit_code)
