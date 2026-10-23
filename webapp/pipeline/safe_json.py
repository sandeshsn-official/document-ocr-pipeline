"""Cross-process locking and atomic JSON writes for the OCR pipeline.

This module does not run OCR stages or modify files unless a function
is explicitly called.
"""

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class LockTimeoutError(TimeoutError):
    """Raised when a file lock cannot be acquired in time."""


@contextmanager
def exclusive_file_lock(target_path, timeout=30.0, poll_interval=0.1):
    """Acquire a cross-process lock associated with target_path.

    Uses exclusive lock-file creation, which works on Windows and Linux.
    The lock file is intentionally retained after release; its existence
    alone does not indicate an active lock.
    """
    target = Path(target_path).resolve()
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout
    acquired = False

    while not acquired:
        try:
            fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(token)
                lock_file.flush()
                os.fsync(lock_file.fileno())
            acquired = True
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"Timed out waiting for lock: {lock_path}"
                )
            time.sleep(poll_interval)

    try:
        yield
    finally:
        # Only remove a lock that still belongs to this process invocation.
        try:
            current_token = lock_path.read_text(encoding="utf-8")
            if current_token == token:
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def atomic_write_json(path, data):
    """Atomically replace a JSON file after writing complete content."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target.with_name(
        f".{target.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, target)

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def update_json_safely(path, updater, timeout=30.0):
    """Lock a JSON file, update its contents, and atomically replace it.

    updater receives the current JSON value, or None if the file is absent.
    If updater raises an exception, the original file remains unchanged.
    """
    target = Path(path).resolve()

    with exclusive_file_lock(target, timeout=timeout):
        if target.exists():
            with target.open("r", encoding="utf-8") as file:
                current_data = json.load(file)
        else:
            current_data = None

        updated_data = updater(current_data)
        atomic_write_json(target, updated_data)

        return updated_data
