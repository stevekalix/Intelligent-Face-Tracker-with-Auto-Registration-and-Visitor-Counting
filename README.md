# Intelligent Face Tracker with Auto-Registration and Visitor Counting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-green.svg)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace-orange.svg)](https://github.com/deepinsight/insightface)
[![Tracker](https://img.shields.io/badge/Tracking-ByteTrack-red.svg)](https://github.com/ifzhang/ByteTrack)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)

---

## 1. Project Title & Hackathon Context
**Intelligent Face Tracker with Auto-Registration and Visitor Counting** is a computer vision and edge-AI surveillance system engineered for the Katamoran Hackathon. 

In automated retail analytics, event security, and smart building access control, traditional visitor counters (such as tripwires or infrared beams) suffer from critical inaccuracies: they fail to distinguish between different individuals, inflate metrics due to staff passing multiple times, and lack visual audit evidence. 

This project solves these limitations by implementing an **end-to-end edge AI pipeline** that:
1. Detects human faces in real time using **Ultralytics YOLOv8**.
2. Extracts 512-dimensional biometric feature embeddings using **InsightFace ArcFace** (strictly avoiding the legacy `face_recognition` library).
3. Maintains temporal trajectory continuity with **ByteTrack** two-stage Hungarian Kalman tracking.
4. Automatically registers first-time visitors into an ACID-compliant **SQLite database**.
5. Employs vectorized cosine similarity matching to ensure that returning visitors **never inflate** the unique visitor count.
6. Enforces strict entry and exit event accounting (exactly one entry and one exit log per person per appearance session), archiving cropped facial snapshots in structured folders (`logs/entries/YYYY-MM-DD/`) and recording comprehensive system telemetry in `events.log`.

---

## 2. System Architecture & Data Flow

```
[Camera / RTSP / MP4] 
        │
        ▼
[Frame Decider] ───(Skip Frame)───► [Kalman Filter Motion Predict] ──┐
        │                                                            │
  (Detect Frame)                                                     │
        ▼                                                            │
[YOLOv8 Face Detection]                                              │
        │                                                            │
        ▼                                                            │
[ByteTrack 2-Stage Matching] ◄───────────────────────────────────────┘
        │
        ▼
[InsightFace ArcFace (512-D Embedding)]
        │
        ▼
[Vectorized Cosine Similarity Search]
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
[Existing Identity]            [Unregistered Identity]
 • Re-identify Person ID        • Generate VISITOR_XXXX ID
 • Visitor count UNCHANGED      • Unique Count INCREMENTS
 • Update last_seen in DB       • Auto-Register in SQLite
   │                                  │
   └───────────────┬──────────────────┘
                   ▼
       [Session State Machine]
      ┌────────────┴────────────┐
      ▼                         ▼
 [ENTRY EVENT]             [EXIT EVENT]
  • Save face crop to:      • Compute duration
    logs/entries/YYYY-MM-DD/• Log EXIT to SQLite
  • Log ENTRY to SQLite     • Log EXIT to events.log
  • Log to events.log
                   │
                   ▼
     [Interactive Heads-Up Display]
      • Live 30+ FPS video canvas
      • Color-coded bounding boxes & badges
      • Total Unique Visitors & Live Occupancy
```

---

## 3. Assumptions Made During Development

1. **Facial Visibility & Resolution**:
   - The camera is assumed to be mounted at an elevation and angle where subjects' faces are at least $30 \times 30$ pixels in bounding box dimension with pitch and yaw within $\pm 45^\circ$.
2. **Session Exit Horizon (`max_lost_frames`)**:
   - A visitor appearance session is considered terminated ("EXIT") when their track has been consecutively absent/unmatched for $N$ frames (default: `30` frames $\approx 1.0$ second at 30 FPS). If the subject re-enters the field of view after this window, a new session is initiated with an `ENTRY` log, but their persistent identity is re-matched against the database and the unique visitor count **does not increase**.
3. **Biometric Similarity Threshold**:
   - ArcFace feature vectors reside on a 512-dimensional hypersphere ($\|\mathbf{e}\|_2 = 1.0$). A cosine similarity cutoff of `0.50` (configurable in `config.json`) provides an optimal balance between False Accept Rate (FAR $\le 0.001\%$) and False Reject Rate (FRR) across varied lighting conditions.
4. **Frame Skipping & Tracking Continuity**:
   - The `frame_skip` parameter (default `2`) assumes that subject velocity between skipped frames is linear, which is accurately handled by the constant-velocity Kalman Filter.
5. **Database Concurrency**:
   - SQLite is utilized for zero-configuration hackathon execution. The database handler utilizes connection pooling with `check_same_thread=False` and indexed queries to ensure non-blocking I/O during 30 FPS stream processing.

---

## 4. Project Structure

```
katamoran/
├── planning.md               # Complete AI Planning & Architecture Document
├── requirements.txt          # Python dependencies
├── config.json               # System configuration parameters
├── db_handler.py             # SQLite database management module
├── logger.py                 # Structured file logging & face crop archival
├── face_processor.py         # YOLOv8 face detector & InsightFace biometrics
├── tracker.py                # ByteTrack multi-object tracker with Kalman filter
├── main.py                   # Main executable pipeline & interactive HUD
├── create_sample_video.py    # Synthetic video generator for instant testing
├── README.md                 # Complete system documentation
├── visitor_system.db         # Generated SQLite database file
├── events.log                # Generated audit ledger file
└── logs/
    └── entries/
        └── YYYY-MM-DD/       # Cropped face images saved on entry
            └── person_VISITOR_0001_143015_123.jpg
```

---

## 5. Setup & Installation Instructions

### 5.1 Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: Version 3.10 or higher
- **Hardware**: CPU (Intel/AMD 4+ cores) or NVIDIA GPU (CUDA 11.8+ / 12.x recommended for maximum throughput)

### 5.2 Step-by-Step Installation

#### 1. Clone the Repository & Navigate to Workspace
```bash
git clone <repository_url>
cd katamoran
```

#### 2. Create and Activate a Python Virtual Environment
*On Windows (PowerShell):*
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*On Linux / macOS:*
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for NVIDIA GPU Acceleration:**
> If you have an NVIDIA GPU with CUDA installed, uninstall the CPU onnxruntime and install the GPU version:
> ```bash
> pip uninstall -y onnxruntime
> pip install onnxruntime-gpu
> ```

---

## 6. Configuration Guide (`config.json`)

All runtime options are managed centrally via `config.json`:

```json
{
  "video_source": "sample.mp4",
  "detector": {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.50,
    "iou_threshold": 0.45,
    "input_size": 640,
    "device": "auto"
  },
  "recognizer": {
    "model_name": "buffalo_s",
    "similarity_threshold": 0.50,
    "embedding_size": 512,
    "device": "auto"
  },
  "tracker": {
    "tracker_type": "bytetrack",
    "track_thresh": 0.50,
    "high_thresh": 0.60,
    "match_thresh": 0.80,
    "track_buffer": 30,
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
    "show_ui": true,
    "window_name": "Intelligent Face Tracker & Visitor Counting",
    "window_width": 1280,
    "window_height": 720,
    "save_output_video": false,
    "output_video_path": "output_annotated.mp4"
  }
}
```

### Switching Video Sources:
- **Local MP4 File**: `"video_source": "sample.mp4"`
- **Live USB Webcam**: `"video_source": 0`
- **RTSP Network Security Camera**: `"video_source": "rtsp://admin:password123@192.168.1.100:554/h264Preview_01_main"`

---

## 7. How to Run

### Step 1: Generate or Provide a Video
If you don't have a test video file immediately available, generate a synthetic multi-subject test clip with one command:
```bash
python create_sample_video.py
```
This generates `sample.mp4` with simulated subjects entering, crossing, exiting, and re-entering to test anti-double counting.

### Step 2: Start the System
```bash
python main.py
```

Or pass a custom video source directly via the command line:
```bash
python main.py --source my_cctv_footage.mp4
# OR
python main.py --source rtsp://user:pass@192.168.1.50:554/stream1
```

### Keyboard Shortcuts during Execution:
- Press **`q`** or **`ESC`**: Gracefully terminate the pipeline, flush exit events, and close the database.
- Press **`s`**: Save an instant high-resolution screenshot with annotations.

---

## 8. Verification & Audit Logs

### 8.1 SQLite Database (`visitor_system.db`)
Inspect registered identities and entry/exit event logs using the SQLite CLI or DB Browser for SQLite:

```sql
-- View all registered unique visitors:
SELECT person_id, first_seen, last_seen, total_appearances, sample_image_path FROM visitors;

-- View chronologically logged entry/exit events:
SELECT person_id, event_type, timestamp, confidence, image_path, notes FROM events ORDER BY id ASC;
```

### 8.2 Event Ledger (`events.log`)
Inspect `events.log` for real-time audit trails:
```
2026-10-16 14:02:10 [INFO] [FaceProcessor]: Loading YOLOv8 detector: yolov8n.pt
2026-10-16 14:02:11 [INFO] [FaceProcessor]: InsightFace recognition engine (ArcFace 512-D) ready.
2026-10-16 14:02:12 [INFO] [FaceTracker]: [ENTRY] Person: VISITOR_0001 | Status: NEW_REGISTRATION | Time: 2026-10-16 14:02:12 | Conf: 0.88 | Image: logs\entries\2026-10-16\person_VISITOR_0001_140212_412.jpg
2026-10-16 14:02:15 [INFO] [FaceTracker]: [RECOGNITION] Track ID: 3 recognized as VISITOR_0001 | Similarity Score: 0.8924
2026-10-16 14:02:18 [INFO] [FaceTracker]: [EXIT] Person: VISITOR_0001 | Time: 2026-10-16 14:02:18 | Session Duration: 6.2 seconds
```

---

## 9. Demonstration Video
Loom or YouTube video link: [Insert Link Here]

---

This project is a part of a hackathon run by https://katomaran.com
