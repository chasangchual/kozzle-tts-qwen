"""Isolated TTS worker subprocess.

The parent process talks to this worker via a JSON-per-line protocol on
stdin/stdout. mlx-audio (and its dependencies: transformers, tqdm, etc.)
write progress and log messages to stdout, which would corrupt the protocol.
To prevent that, we dup the original stdout to a private fd used only for
JSON, then redirect fd 1 (stdout) to fd 2 (stderr). Any library that writes
to stdout — Python-level or C-level — ends up on stderr, which the parent
inherits so the user can see it.
"""

import json
import os
import sys
from typing import IO


def _redirect_stdout_to_stderr() -> IO[str]:
    """Move JSON channel off fd 1 so library prints don't pollute it.

    Returns a Python file object writing to the original stdout fd, intended
    for protocol responses only.
    """
    original_stdout_fd = os.dup(sys.stdout.fileno())
    # Make sure anything previously buffered in Python's stdout gets out
    # before we swap the underlying fd.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # Replace sys.stdout so accidental ``print(...)`` calls in this process
    # also land on stderr.
    sys.stdout = sys.stderr  # type: ignore[assignment]
    return os.fdopen(original_stdout_fd, "w", buffering=1, encoding="utf-8")


_PROTOCOL_OUT = _redirect_stdout_to_stderr()


def _send(msg: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(msg) + "\n")
    _PROTOCOL_OUT.flush()


def _recv() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def main() -> None:
    # Imported lazily so any import-time prints land on stderr (fd 1 already
    # points to stderr by the time this runs).
    from kozzle_tts.config import TTSConfig
    from kozzle_tts.tts import TTSModel

    model: TTSModel | None = None

    while True:
        try:
            request = _recv()
        except Exception as e:
            _send({"status": "error", "message": f"bad request: {e}"})
            continue
        if request is None:
            break

        cmd = request.get("cmd")

        if cmd == "load":
            try:
                config = TTSConfig(**request["config"])
                model = TTSModel()
                model.load(config)
                _send({"status": "ready"})
            except Exception as e:
                _send({"status": "error", "message": str(e)})

        elif cmd == "generate":
            if model is None:
                _send({"status": "error", "message": "Model not loaded"})
                continue
            try:
                from pathlib import Path

                config = TTSConfig(**request["config"])
                output_path = request["output_path"]
                result = model.generate_audio(
                    request["text"],
                    Path(output_path),
                    config,
                )
                _send({"status": "ok", "path": str(result)})
            except Exception as e:
                _send({"status": "error", "message": str(e)})

        elif cmd == "shutdown":
            break


if __name__ == "__main__":
    main()
