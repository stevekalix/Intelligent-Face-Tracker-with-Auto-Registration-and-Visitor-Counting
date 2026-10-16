"""
Sample Video Generator Utility.
Generates a synthetic 1080p test video ('sample.mp4') simulating visitors entering,
moving across the field of view, exiting, and returning.
Ensures zero-friction testing out of the box during hackathon demonstrations.
"""

import os
import cv2
import numpy as np


def draw_synthetic_face(frame, center_x, center_y, scale=1.0, face_color=(200, 220, 240)):
    """Draw a stylized humanoid facial representation with features."""
    r = int(50 * scale)
    cx, cy = int(center_x), int(center_y)

    # Head oval
    cv2.ellipse(frame, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, face_color, -1)
    cv2.ellipse(frame, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, (50, 50, 50), 2)

    # Eyes
    eye_offset_x = int(r * 0.40)
    eye_offset_y = int(r * 0.25)
    cv2.circle(frame, (cx - eye_offset_x, cy - eye_offset_y), int(r * 0.12), (255, 255, 255), -1)
    cv2.circle(frame, (cx + eye_offset_x, cy - eye_offset_y), int(r * 0.12), (255, 255, 255), -1)
    cv2.circle(frame, (cx - eye_offset_x, cy - eye_offset_y), int(r * 0.06), (0, 0, 0), -1)
    cv2.circle(frame, (cx + eye_offset_x, cy - eye_offset_y), int(r * 0.06), (0, 0, 0), -1)

    # Eyebrows
    cv2.line(frame, (cx - eye_offset_x - 10, cy - eye_offset_y - 12), (cx - eye_offset_x + 10, cy - eye_offset_y - 10), (40, 40, 40), 2)
    cv2.line(frame, (cx + eye_offset_x - 10, cy - eye_offset_y - 10), (cx + eye_offset_x + 10, cy - eye_offset_y - 12), (40, 40, 40), 2)

    # Nose
    cv2.line(frame, (cx, cy - 5), (cx - 4, cy + 12), (100, 100, 100), 2)
    cv2.line(frame, (cx - 4, cy + 12), (cx + 6, cy + 12), (100, 100, 100), 2)

    # Mouth (Smile)
    cv2.ellipse(frame, (cx, cy + int(r * 0.45)), (int(r * 0.35), int(r * 0.18)), 0, 0, 180, (50, 50, 180), 2)


def generate_sample_video(output_filename="sample.mp4", duration_sec=12, fps=30):
    """Generate a synthetic 1080p video with multiple moving synthetic faces."""
    width, height = 1280, 720
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"[INFO] Generating '{output_filename}' ({width}x{height} @ {fps} FPS, {duration_sec}s)...")

    for f in range(total_frames):
        # Create subtle gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(25, 45, width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(30, 50, width, dtype=np.uint8)
        frame[:, :, 2] = np.linspace(35, 55, width, dtype=np.uint8)

        # Subtle floor grid
        for gy in range(height // 2, height, 40):
            cv2.line(frame, (0, gy), (width, gy), (45, 55, 65), 1)

        # Subject 1: Enters from left, moves to center-right, then exits
        # Active from frame 0 to frame 220
        if 0 <= f < 220:
            s1_x = int(100 + (f / 220.0) * (width - 200))
            s1_y = int(320 + np.sin(f * 0.05) * 20)
            draw_synthetic_face(frame, s1_x, s1_y, scale=1.3, face_color=(190, 215, 235))
            cv2.putText(frame, "Synthetic Face A", (s1_x - 60, s1_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Subject 2: Enters from right, crosses towards left
        # Active from frame 70 to frame 300
        if 70 <= f < 300:
            rel_f = f - 70
            s2_x = int((width - 150) - (rel_f / 230.0) * (width - 300))
            s2_y = int(400 + np.cos(rel_f * 0.04) * 25)
            draw_synthetic_face(frame, s2_x, s2_y, scale=1.2, face_color=(185, 230, 210))
            cv2.putText(frame, "Synthetic Face B", (s2_x - 60, s2_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Re-entry Test: Subject 1 returns at frame 260 to test anti-double counting!
        if f >= 260:
            rel_f = f - 260
            s1_x = int(120 + (rel_f / 100.0) * 300)
            s1_y = int(340 + np.sin(rel_f * 0.06) * 15)
            draw_synthetic_face(frame, s1_x, s1_y, scale=1.3, face_color=(190, 215, 235))
            cv2.putText(frame, "Synthetic Face A (Returning)", (s1_x - 90, s1_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)

        # On-screen guidance banner
        cv2.putText(frame, f"Test Video Simulation | Frame: {f}/{total_frames}", (25, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample test video saved: {output_filename}")


if __name__ == "__main__":
    generate_sample_video()
