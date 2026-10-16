"""
Main Application Pipeline: Intelligent Face Tracker with Auto-Registration and Visitor Counting.

Orchestrates:
1. Video Capture (Local MP4 or RTSP stream from config.json)
2. Frame Skipping (Heavy detection/embedding vs lightweight Kalman tracking)
3. YOLOv8 Face Detection
4. InsightFace ArcFace 512-D Biometrics
5. ByteTrack Multi-Object Tracking
6. Auto-Registration Engine (SQLite Database)
7. Anti-Double-Counting Logic
8. Entry/Exit Event Logging & Face Crop Archival
9. Interactive Heads-Up Display (HUD)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional
import cv2
import numpy as np

# Internal system modules
from db_handler import DatabaseHandler
from logger import (
    setup_logger,
    save_cropped_face,
    log_entry,
    log_exit,
    log_recognition,
    log_tracking,
    log_embedding_generation
)
from face_processor import FaceProcessor
from tracker import ByteTracker, TrackState, Track


def load_config(config_path: str = "config.json") -> dict:
    """Load system configuration from JSON file with fallback defaults."""
    default_config = {
        "video_source": "sample.mp4",
        "detector": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.50,
            "iou_threshold": 0.45,
            "device": "auto"
        },
        "recognizer": {
            "model_name": "buffalo_s",
            "similarity_threshold": 0.50,
            "device": "auto"
        },
        "tracker": {
            "track_thresh": 0.50,
            "high_thresh": 0.60,
            "match_thresh": 0.80,
            "max_lost_frames": 30,
            "frame_skip": 2
        },
        "database": {
            "db_path": "visitor_system.db"
        },
        "logging": {
            "log_file": "events.log",
            "entries_dir": "logs/entries",
            "log_level": "INFO"
        },
        "display": {
            "show_ui": True,
            "window_name": "Intelligent Face Tracker & Visitor Counting",
            "window_width": 1280,
            "window_height": 720,
            "save_output_video": False,
            "output_video_path": "output_annotated.mp4"
        }
    }

    if not os.path.exists(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Using default parameters.")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            # Recursive update
            for key, val in user_config.items():
                if isinstance(val, dict) and key in default_config:
                    default_config[key].update(val)
                else:
                    default_config[key] = val
        return default_config
    except Exception as e:
        print(f"[ERROR] Failed to parse '{config_path}': {e}. Using defaults.")
        return default_config


def draw_hud_overlay(
    frame: np.ndarray,
    unique_count: int,
    in_frame_count: int,
    fps: float,
    frame_skip: int,
    is_detection_frame: bool
) -> np.ndarray:
    """Render a modern visual Heads-Up Display (HUD) banner on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent HUD Header background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Accent top border line
    cv2.line(frame, (0, 64), (w, 64), (0, 210, 255), 2)

    # 1. Title
    cv2.putText(
        frame,
        "KATAMORAN AI | VISITOR INTELLIGENCE SYSTEM",
        (20, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Mode indicator
    mode_text = "MODE: DETECTION CYCLE" if is_detection_frame else "MODE: KALMAN TRACKING"
    mode_color = (0, 255, 128) if is_detection_frame else (200, 200, 200)
    cv2.putText(
        frame,
        f"{mode_text} (Skip: {frame_skip})",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        mode_color,
        1,
        cv2.LINE_AA
    )

    # 2. Metric Pills (Right aligned)
    # Total Unique Visitors
    pill1_x = w - 480
    cv2.rectangle(frame, (pill1_x, 12), (pill1_x + 190, 52), (30, 40, 50), -1)
    cv2.rectangle(frame, (pill1_x, 12), (pill1_x + 190, 52), (0, 210, 255), 1)
    cv2.putText(frame, "UNIQUE VISITORS", (pill1_x + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{unique_count}", (pill1_x + 10, 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Currently In Frame
    pill2_x = w - 275
    cv2.rectangle(frame, (pill2_x, 12), (pill2_x + 140, 52), (30, 40, 50), -1)
    cv2.rectangle(frame, (pill2_x, 12), (pill2_x + 140, 52), (0, 255, 128), 1)
    cv2.putText(frame, "IN SCENE", (pill2_x + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{in_frame_count}", (pill2_x + 10, 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)

    # Real-time FPS
    pill3_x = w - 120
    cv2.rectangle(frame, (pill3_x, 12), (pill3_x + 100, 52), (30, 40, 50), -1)
    cv2.rectangle(frame, (pill3_x, 12), (pill3_x + 100, 52), (220, 100, 0), 1)
    cv2.putText(frame, "FPS", (pill3_x + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f}", (pill3_x + 10, 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def render_track_annotations(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draw stylish bounding boxes and identification badges around tracked faces."""
    for track in tracks:
        x1, y1, x2, y2 = track.bbox

        # Palette: Cyan/Green for confirmed registered identities, Orange for pending
        if track.person_id:
            box_color = (0, 220, 100)  # Greenish
            badge_title = f"{track.person_id}"
        else:
            box_color = (0, 180, 255)  # Amber / Cyan
            badge_title = f"Track #{track.track_id}"

        # Bounding box corners styling
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Corner accents
        line_len = max(10, min(20, (x2 - x1) // 4))
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), (255, 255, 255), 3)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), (255, 255, 255), 3)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), (255, 255, 255), 3)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), (255, 255, 255), 3)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), (255, 255, 255), 3)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), (255, 255, 255), 3)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), (255, 255, 255), 3)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), (255, 255, 255), 3)

        # Label Pill above box
        label_text = f"{badge_title} | {track.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        badge_y1 = max(0, y1 - text_h - 12)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = x1 + text_w + 14

        # Pill background
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
        # Pill text
        cv2.putText(
            frame,
            label_text,
            (badge_x1 + 7, badge_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 15, 20),
            1,
            cv2.LINE_AA
        )

    return frame


def run_pipeline(config_path: str = "config.json") -> None:
    """Execute the end-to-end intelligent face tracking and visitor counting pipeline."""
    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Setup Logging
    log_cfg = config.get("logging", {})
    logger = setup_logger(
        log_file=log_cfg.get("log_file", "events.log"),
        log_level=log_cfg.get("log_level", "INFO"),
        logger_name="MainPipeline"
    )
    logger.info("=" * 70)
    logger.info("STARTING INTELLIGENT FACE TRACKER & VISITOR COUNTING PIPELINE")
    logger.info("=" * 70)

    # 3. Initialize Database (MySQL or SQLite)
    db_cfg = config.get("database", {})
    db = DatabaseHandler(
        db_type=db_cfg.get("db_type", "sqlite"),
        db_path=db_cfg.get("db_path", "visitor_system.db"),
        mysql_config=db_cfg.get("mysql", None)
    )

    # Load known embeddings into memory cache
    known_embeddings: Dict[str, np.ndarray] = db.get_all_embeddings()
    unique_visitor_count: int = db.get_unique_visitor_count()
    logger.info("Loaded %d registered visitor identities from database.", unique_visitor_count)

    # 4. Initialize FaceProcessor (YOLOv8 + InsightFace)
    det_cfg = config.get("detector", {})
    rec_cfg = config.get("recognizer", {})
    processor = FaceProcessor(
        yolo_model_path=det_cfg.get("model_path", "yolov8n.pt"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.50),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        insightface_model_name=rec_cfg.get("model_name", "buffalo_s"),
        similarity_threshold=rec_cfg.get("similarity_threshold", 0.50),
        device=det_cfg.get("device", "auto")
    )

    # 5. Initialize ByteTracker
    trk_cfg = config.get("tracker", {})
    frame_skip = int(trk_cfg.get("frame_skip", 2))
    tracker = ByteTracker(
        track_thresh=trk_cfg.get("track_thresh", 0.50),
        high_thresh=trk_cfg.get("high_thresh", 0.60),
        match_thresh=trk_cfg.get("match_thresh", 0.80),
        max_lost_frames=trk_cfg.get("max_lost_frames", 30)
    )

    # 6. Open Video Stream (File or RTSP)
    video_source = config.get("video_source", "sample.mp4")
    # Convert string integer (like "0") to int for webcam
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    logger.info("Attempting to connect to video source: %s", video_source)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", video_source)
        if isinstance(video_source, str) and video_source.endswith(".mp4"):
            logger.info("Tip: Place your video file at '%s' or run 'python create_sample_video.py' to generate a test clip.", video_source)
        sys.exit(1)

    fps_stream = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    logger.info("Stream opened successfully. Resolution: %dx%d | FPS: %.1f | Total Frames: %d",
                width, height, fps_stream, total_frames)

    # Display / Output Video Setup
    disp_cfg = config.get("display", {})
    show_ui = disp_cfg.get("show_ui", True)
    window_name = disp_cfg.get("window_name", "Intelligent Face Tracker & Visitor Counting")
    save_video = disp_cfg.get("save_output_video", False)
    out_video_path = disp_cfg.get("output_video_path", "output_annotated.mp4")
    video_writer = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps_stream, (width, height))
        logger.info("Saving output video to: %s", out_video_path)

    if show_ui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_cfg.get("window_width", 1280), disp_cfg.get("window_height", 720))

    # Runtime State Variables
    frame_idx = 0
    fps_estimate = fps_stream
    prev_time = time.perf_counter()
    active_in_frame_tracks = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Stream finished or reached end of video file. Exiting main loop.")
                break

            t_frame_start = time.perf_counter()
            is_detection_frame = (frame_idx % (frame_skip + 1) == 0)

            # -----------------------------------------------------------------
            # BRANCH A: HEAVY DETECTION & RECOGNITION FRAME
            # -----------------------------------------------------------------
            if is_detection_frame:
                # 1. Detect faces using YOLOv8
                detections = processor.detect_faces(frame)

                # 2. Update ByteTrack tracker
                active_in_frame_tracks = tracker.update(detections)

                # 3. Biometric Matching & Auto-Registration
                for track in active_in_frame_tracks:
                    # If this track does not yet have a persistent identity
                    if track.person_id is None:
                        t_emb_start = time.perf_counter()
                        # Extract embedding from face crop or frame bbox
                        emb = processor.extract_embedding(frame, bbox=track.bbox, face_crop=track.best_crop)
                        duration_ms = (time.perf_counter() - t_emb_start) * 1000

                        if emb is not None:
                            track.embedding = emb
                            log_embedding_generation(f"Track_{track.track_id}", duration_ms)

                            # Match against all known registered embeddings
                            matched_id, score = processor.find_matching_visitor(emb, known_embeddings)

                            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            if matched_id is not None:
                                # RECOGNIZED EXISTING VISITOR
                                track.person_id = matched_id
                                log_recognition(matched_id, score, track.track_id)
                                db.update_visitor_seen(matched_id, timestamp_now)

                                # Save entry image and log ENTRY if not yet logged for this session
                                if not track.entry_logged:
                                    img_path = save_cropped_face(track.best_crop, matched_id, timestamp_now, base_dir=log_cfg.get("entries_dir", "logs/entries"))
                                    db.log_event(matched_id, "ENTRY", timestamp_now, confidence=score, image_path=img_path)
                                    log_entry(matched_id, timestamp_now, img_path, confidence=score, is_new_registration=False)
                                    track.entry_logged = True
                            else:
                                # UNRECOGNIZED: AUTO-REGISTER NEW VISITOR
                                new_id = db.get_next_visitor_id()
                                track.person_id = new_id

                                # Save face crop to logs/entries/YYYY-MM-DD/
                                img_path = save_cropped_face(track.best_crop, new_id, timestamp_now, base_dir=log_cfg.get("entries_dir", "logs/entries"))

                                # Register in SQLite
                                db.register_visitor(new_id, emb, image_path=img_path, timestamp=timestamp_now)
                                db.log_event(new_id, "ENTRY", timestamp_now, confidence=track.score, image_path=img_path)

                                # Update memory cache
                                known_embeddings[new_id] = emb
                                unique_visitor_count = db.get_unique_visitor_count()

                                # Log to events.log
                                log_entry(new_id, timestamp_now, img_path, confidence=track.score, is_new_registration=True)
                                track.entry_logged = True
                                logger.info(">> NEW VISITOR AUTO-REGISTERED: %s | Total Count: %d", new_id, unique_visitor_count)

                    else:
                        # Known identity already associated with track; ensure entry is logged if missed
                        if not track.entry_logged:
                            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            img_path = save_cropped_face(track.best_crop, track.person_id, timestamp_now, base_dir=log_cfg.get("entries_dir", "logs/entries"))
                            db.log_event(track.person_id, "ENTRY", timestamp_now, confidence=track.score, image_path=img_path)
                            log_entry(track.person_id, timestamp_now, img_path, confidence=track.score, is_new_registration=False)
                            track.entry_logged = True

            # -----------------------------------------------------------------
            # BRANCH B: SKIPPED FRAME (LIGHTWEIGHT KALMAN PREDICTION)
            # -----------------------------------------------------------------
            else:
                # Advance Kalman filter predictions without running YOLOv8 or InsightFace
                active_in_frame_tracks = tracker.predict_skipped_frame()

            # -----------------------------------------------------------------
            # 4. EXIT DETECTION & LOGGING
            # -----------------------------------------------------------------
            # Retrieve tracks that have exited the frame (lost for > max_lost_frames)
            exited_tracks = tracker.get_newly_exited_tracks()
            for exit_track in exited_tracks:
                if exit_track.person_id is not None:
                    exit_time = datetime.now()
                    exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
                    duration_sec = (exit_time - exit_track.start_time).total_seconds()

                    # Write EXIT event to SQLite DB
                    db.log_event(
                        exit_track.person_id,
                        "EXIT",
                        exit_time_str,
                        confidence=1.0,
                        notes=f"Session Duration: {duration_sec:.1f}s"
                    )

                    # Write EXIT event to events.log
                    log_exit(exit_track.person_id, exit_time_str, duration_sec)
                    log_tracking(exit_track.track_id, "EXITED_SCENE", f"Person: {exit_track.person_id}")

            # -----------------------------------------------------------------
            # 5. VISUAL RENDERING & DISPLAY
            # -----------------------------------------------------------------
            # Draw bounding boxes and identification labels
            render_track_annotations(frame, active_in_frame_tracks)

            # Compute smoothed FPS
            dt = time.perf_counter() - t_frame_start
            instant_fps = 1.0 / max(dt, 1e-4)
            fps_estimate = 0.90 * fps_estimate + 0.10 * instant_fps

            # Render top HUD dashboard
            draw_hud_overlay(
                frame,
                unique_count=unique_visitor_count,
                in_frame_count=len(active_in_frame_tracks),
                fps=fps_estimate,
                frame_skip=frame_skip,
                is_detection_frame=is_detection_frame
            )

            # Save annotated output video frame if configured
            if video_writer is not None:
                video_writer.write(frame)

            # Display GUI window
            if show_ui:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    logger.info("User requested exit ('q' key pressed).")
                    break
                elif key == ord("s"):  # Save screenshot
                    snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snap_name, frame)
                    logger.info("Saved screenshot to %s", snap_name)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating gracefully...")
    except Exception as e:
        logger.exception("Unexpected error in main processing loop: %s", e)
    finally:
        # Flush exit events for any remaining active tracks
        logger.info("Flushing exit logs for active tracks before system shutdown...")
        cleanup_time = datetime.now()
        cleanup_time_str = cleanup_time.strftime("%Y-%m-%d %H:%M:%S")

        for track in active_in_frame_tracks:
            if track.person_id and not track.exit_logged:
                duration_sec = (cleanup_time - track.start_time).total_seconds()
                db.log_event(track.person_id, "EXIT", cleanup_time_str, confidence=1.0, notes="Stream terminated")
                log_exit(track.person_id, cleanup_time_str, duration_sec)
                track.exit_logged = True

        # Resource clean-up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_ui:
            cv2.destroyAllWindows()
        db.close()
        logger.info("Pipeline terminated successfully. All records persisted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker with Auto-Registration and Visitor Counting")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--source", type=str, default=None, help="Override video source (video file or RTSP stream)")
    args = parser.parse_args()

    # Allow command line override of video source
    if args.source:
        cfg = load_config(args.config)
        cfg["video_source"] = args.source
        with open(args.config, "w", encoding="utf-8") as f_out:
            json.dump(cfg, f_out, indent=2)

    run_pipeline(args.config)
