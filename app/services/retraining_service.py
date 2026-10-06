import os
import subprocess
import sys
import threading

from app.db.database import SessionLocal
from app.db.repository import (
    get_untrained_dislike_ids,
    mark_feedback_used_for_training,
)
from ml.model_registry import get_active_model_version

from app.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RETRAINING_LOCK_PATH,
    RETRAINING_LOG_PATH,
)


_retraining_process = None


def _is_process_running(pid: int) -> bool:

    try:
        os.kill(pid, 0)

    except OSError:
        return False

    return True


def _acquire_lock() -> bool:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RETRAINING_LOCK_PATH.exists():

        try:

            pid = int(
                RETRAINING_LOCK_PATH.read_text(
                    encoding="utf-8"
                ).strip()
            )

        except (ValueError, OSError):

            RETRAINING_LOCK_PATH.unlink(
                missing_ok=True
            )

        else:

            if _is_process_running(pid):
                return False

            RETRAINING_LOCK_PATH.unlink(
                missing_ok=True
            )

    RETRAINING_LOCK_PATH.write_text(
        "starting",
        encoding="utf-8",
    )

    return True


def _release_lock() -> None:

    RETRAINING_LOCK_PATH.unlink(
        missing_ok=True
    )


def _monitor_retraining(
    process,
    model_manager,
    previous_active_version: str,
    feedback_ids: list[int],
) -> None:

    global _retraining_process

    try:

        return_code = process.wait()

        print(
            f"[RETRAIN] Training process finished "
            f"with return code: {return_code}"
        )

        if return_code != 0:

            print(
                "[RETRAIN] Training failed. "
                "Current model will remain active."
            )

            return

        current_active_version = get_active_model_version()

        print(
            f"[RETRAIN] Previous active model: "
            f"{previous_active_version}"
        )

        print(
            f"[RETRAIN] Current active model: "
            f"{current_active_version}"
        )

        if current_active_version == previous_active_version:

            print(
                "[RETRAIN] New model was not promoted. "
                "Keeping feedback available for future training."
            )

            return

        print(
            "[RETRAIN] New model was accepted and promoted."
        )

        print(
            f"[RETRAIN] Marking {len(feedback_ids)} "
            "feedback records as used for training."
        )

        with SessionLocal() as db:

            mark_feedback_used_for_training(
                db=db,
                feedback_ids=feedback_ids,
            )

        print(
            "[RETRAIN] Feedback records marked as used."
        )

        print(
            "[RETRAIN] Reloading ModelManager..."
        )

        model_manager.reload_model()

        print(
            "[RETRAIN] ModelManager reloaded successfully."
        )

    except Exception as exc:

        print(
            f"[RETRAIN] Error while finalizing retraining: "
            f"{exc}"
        )

    finally:

        _release_lock()

        _retraining_process = None

        print(
            "[RETRAIN] Retraining cleanup completed."
        )


def start_retraining(model_manager) -> bool:

    global _retraining_process

    print("[RETRAIN] start_retraining() called")

    if not _acquire_lock():

        print(
            "[RETRAIN] Retraining already in progress"
        )

        return False

    try:

        # Capture the feedback that triggered this run.
        with SessionLocal() as db:

            feedback_ids = get_untrained_dislike_ids(
                db
            )

        print(
            f"[RETRAIN] Feedback records included "
            f"in this training cycle: {feedback_ids}"
        )

        # Remember which model was active before training.
        previous_active_version = get_active_model_version()

        print(
            f"[RETRAIN] Active model before training: "
            f"{previous_active_version}"
        )

        print(
            f"[RETRAIN] Starting training from: "
            f"{PROJECT_ROOT}"
        )

        log_file = RETRAINING_LOG_PATH.open(
            "a",
            encoding="utf-8",
        )

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ml.train",
            ],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        log_file.close()

        RETRAINING_LOCK_PATH.write_text(
            str(process.pid),
            encoding="utf-8",
        )

        _retraining_process = process

        print(
            f"[RETRAIN] Training process started. "
            f"PID={process.pid}"
        )

        monitor_thread = threading.Thread(
            target=_monitor_retraining,
            args=(
                process,
                model_manager,
                previous_active_version,
                feedback_ids,
            ),
            daemon=True,
        )

        monitor_thread.start()

        print(
            "[RETRAIN] Training monitor started"
        )

        return True

    except Exception:

        _release_lock()

        raise


def is_retraining_in_progress() -> bool:
    global _retraining_process
    
    if _retraining_process is None:
        return False
    return _retraining_process.poll() is None   