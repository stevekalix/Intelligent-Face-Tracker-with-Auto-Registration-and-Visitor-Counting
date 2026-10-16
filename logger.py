"""
Logging System Module for Intelligent Face Tracker & Visitor Counting.
Configures system-wide audit logging to 'events.log' and handles structured local image
storage in date-partitioned folders: logs/entries/YYYY-MM-DD/.
"""

import os
import logging
from datetime import datetime
from typing import Optional
import cv2
import numpy as np


def setup_logger(
    log_file: str = "events.log",
    log_level: str = "INFO",
    logger_name: str = "FaceTracker"
) -> logging.Logger:
    """Set up and configure the primary system logger.

    Writes formatted logs to both standard console output and a local file 'events.log'.

    Args:
        log_file: Path to the output text log file.
        log_level: Logging severity level (DEBUG, INFO, WARNING, ERROR).
        logger_name: Base namespace name for the logger.

    Returns:
        Configured logging.Logger instance.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler (Audit Trail)
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Stream Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Default module-level logger instance
system_logger = setup_logger()


def save_cropped_face(
    face_img: np.ndarray,
    person_id: str,
    timestamp: Optional[str] = None,
    base_dir: str = "logs/entries"
) -> Optional[str]:
    """Save a cropped face image to a date-partitioned folder structure.

    Target directory: logs/entries/YYYY-MM-DD/person_<person_id>_<timestamp>.jpg

    Args:
        face_img: BGR numpy image array of the cropped face.
        person_id: Unique identifier of the visitor.
        timestamp: Timestamp string. If None, current time is used.
        base_dir: Base directory for storing entry crops.

    Returns:
        Normalized relative path of the saved image file, or None if save failed.
    """
    if face_img is None or face_img.size == 0:
        system_logger.warning("Attempted to save an empty face image for [%s].", person_id)
        return None

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S_%f")[:10]  # millisecond precision

    # Construct target folder: logs/entries/YYYY-MM-DD/
    target_folder = os.path.join(base_dir, date_str)
    os.makedirs(target_folder, exist_ok=True)

    # Clean filename
    safe_person_id = str(person_id).replace(" ", "_")
    filename = f"person_{safe_person_id}_{time_str}.jpg"
    file_path = os.path.join(target_folder, filename)

    try:
        success = cv2.imwrite(file_path, face_img)
        if success:
            system_logger.debug("Saved face crop to: %s", file_path)
            return os.path.normpath(file_path)
        else:
            system_logger.error("cv2.imwrite failed to save image to: %s", file_path)
            return None
    except Exception as e:
        system_logger.error("Exception occurred while saving cropped face image: %s", e)
        return None


# =========================================================================
# Structured Event Logging Helpers (Standardized entries in events.log)
# =========================================================================

def log_entry(
    person_id: str,
    timestamp: str,
    image_path: Optional[str],
    confidence: float = 1.0,
    is_new_registration: bool = False
) -> None:
    """Log an ENTRY event to events.log."""
    reg_status = "NEW_REGISTRATION" if is_new_registration else "KNOWN_VISITOR"
    msg = (
        f"[ENTRY] Person: {person_id} | Status: {reg_status} | Time: {timestamp} | "
        f"Conf: {confidence:.2f} | Image: {image_path}"
    )
    system_logger.info(msg)


def log_exit(person_id: str, timestamp: str, duration_seconds: float) -> None:
    """Log an EXIT event to events.log."""
    msg = (
        f"[EXIT] Person: {person_id} | Time: {timestamp} | "
        f"Session Duration: {duration_seconds:.1f} seconds"
    )
    system_logger.info(msg)


def log_recognition(
    person_id: str,
    similarity_score: float,
    track_id: int
) -> None:
    """Log a RECOGNITION event to events.log."""
    msg = (
        f"[RECOGNITION] Track ID: {track_id} recognized as {person_id} | "
        f"Similarity Score: {similarity_score:.4f}"
    )
    system_logger.info(msg)


def log_tracking(
    track_id: int,
    action: str,
    details: Optional[str] = ""
) -> None:
    """Log a TRACKING lifecycle update to events.log."""
    msg = f"[TRACKING] Track ID: {track_id} | Action: {action} | Details: {details}"
    system_logger.debug(msg)


def log_embedding_generation(
    target_id: str,
    duration_ms: float,
    dimensions: int = 512
) -> None:
    """Log an EMBEDDING GENERATION event to events.log."""
    msg = (
        f"[EMBEDDING_GEN] Identity: {target_id} | Dimensions: {dimensions}-D | "
        f"Compute Time: {duration_ms:.2f} ms"
    )
    system_logger.debug(msg)
