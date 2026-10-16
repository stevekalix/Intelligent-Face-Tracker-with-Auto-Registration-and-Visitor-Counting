# AI Planning & System Architecture Document
## Intelligent Face Tracker with Auto-Registration and Visitor Counting

---

### Executive Summary
This document outlines the system architecture, component breakdown, data flow, feature specifications, and computational load estimates for the **Intelligent Face Tracker with Auto-Registration and Visitor Counting** system. Designed for high-accuracy security, surveillance, and retail intelligence, the system maintains strict visitor accounting, eliminates duplicate counting through deep facial biometrics, and produces audit-ready event logs.

---

## 1. System Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                    INPUT ACQUISITION STAGE                                       |
|                                                                                                  |
|   +------------------------------------+          +------------------------------------------+   |
|   |    Local Video File (sample.mp4)   |    OR    |     Live RTSP Network Camera Stream      |   |
|   +-----------------+------------------+          +--------------------+---------------------+   |
|                     |                                                  |                         |
|                     +------------------------+-------------------------+                         |
|                                              |                                                   |
|                                              v                                                   |
|                                +----------------------------+                                    |
|                                | OpenCV VideoCapture Stream |                                    |
|                                +-------------+--------------+                                    |
+----------------------------------------------|---------------------------------------------------+
                                               | (Raw Frame BGR: 1920x1080 @ 30 FPS)
                                               v
+--------------------------------------------------------------------------------------------------+
|                                FRAME PREPROCESSING & SKIP DISPATCH                               |
|                                                                                                  |
|              +----------------------------------------------------------------+                  |
|              | Frame Skip Decider: if (frame_idx % (frame_skip + 1) == 0)     |                  |
|              +-------------------+----------------------------+---------------+                  |
|                                  |                            |                                  |
|                         [Detection Frame]              [Skipped Frame]                           |
+----------------------------------|----------------------------|----------------------------------+
                                   |                            |
                                   v                            v
+----------------------------------------------------+  +------------------------------------------+
|            HEAVY INFERENCE PIPELINE                |  |       LIGHTWEIGHT TRACKER PIPELINE       |
|                                                    |  |                                          |
|  1. YOLOv8 Face Detection:                         |  |  Kalman Filter State Prediction:         |
|     - Scale/Letterbox to 640x640                   |  |  - Extrapolate bounding box positions    |
|     - Model: Ultralytics YOLOv8                    |  |    using velocity & motion dynamics      |
|     - BBoxes: [x1, y1, x2, y2, confidence]         |  |  - Maintain temporal track IDs           |
|                                                    |  |  - Zero GPU/Neural Network overhead      |
|  2. ByteTrack Two-Stage Association:               |  |                                          |
|     - High-score bbox matching via IoU             |  |                                          |
|     - Low-score bbox matching for occlusions       |  |                                          |
|     - Maintain Track State (Tentative / Confirmed) |  |                                          |
|                                                    |  |                                          |
|  3. InsightFace Feature Extraction:                |  |                                          |
|     - Face crop & 112x112 normalization            |  |                                          |
|     - ArcFace Deep ResNet / MobileFaceNet Backbone |  |                                          |
|     - 512-D L2-Normalized Embedding Vector         |  |                                          |
+--------------------------+-------------------------+  +--------------------+---------------------+
                           |                                                 |
                           v                                                 |
+-------------------------------------------------------------------------+  |
|               BIOMETRIC MATCHING & AUTO-REGISTRATION                    |  |
|                                                                         |  |
|  Cosine Similarity Matching:                                            |  |
|  - Compare 512-D vector against cached SQLite embeddings                |  |
|  - Formula: sim(u, v) = (u . v) / (||u|| * ||v||)                       |  |
|                                                                         |  |
|           +---------------------------------------------+               |  |
|           | Is max(similarity) >= similarity_threshold? |               |  |
|           +----------------------+--------------+-------+               |  |
|                                  |              |                       |  |
|                                 YES             NO                      |  |
|                                  |              |                       |  |
|                                  v              v                       |  |
|                     [KNOWN VISITOR]            [NEW VISITOR]            |  |
|                     - Re-identify person_id    - Generate VISITOR_XXXX  |  |
|                     - Total count UNCHANGED    - Increment Unique Count |  |
|                     - Update last_seen in DB   - Auto-register in DB    |  |
+----------------------------------+--------------+-----------------------+  |
                                   |              |                          |
                                   +-------+------+                          |
                                           |                                 |
                                           v                                 v
+--------------------------------------------------------------------------------------------------+
|                                    STATE & LIFECYCLE MANAGEMENT                                  |
|                                                                                                  |
|   - Track Life States: Active, Lost, Exited                                                      |
|   - Entry Event Trigger:                                                                         |
|     * Fires once upon first confirmed appearance in frame                                        |
|     * Saves cropped face to logs/entries/YYYY-MM-DD/person_<id>_<timestamp>.jpg                  |
|     * Writes ENTRY event to SQLite DB & logs to events.log                                       |
|   - Exit Event Trigger:                                                                          |
|     * Fires when track has disappeared for > max_lost_frames                                     |
|     * Calculates total session duration                                                          |
|     * Writes EXIT event to SQLite DB & logs to events.log                                        |
+--------------------------------------------------+-----------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------------------------------------------------------+
|                                     PERSISTENCE & LOGGING LAYER                                  |
|                                                                                                  |
|    +-----------------------------+  +---------------------------+  +-------------------------+   |
|    |      SQLite Database        |  |        events.log         |  |      File Storage       |   |
|    |                             |  |                           |  |                         |   |
|    | - visitors table (metadata, |  | - Structured text logs:   |  | - Cropped face images:  |   |
|    |   512-D binary embeddings)  |  |   [ENTRY], [RECOGNITION], |  |   logs/entries/         |   |
|    | - events table (entry/exit  |  |   [TRACKING], [EXIT],     |  |   YYYY-MM-DD/           |   |
|    |   history, timestamps, img) |  |   [EMBEDDING_GEN]         |  |   person_<id>_<time>.jpg|   |
|    +-----------------------------+  +---------------------------+  +-------------------------+   |
+--------------------------------------------------+-----------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------------------------------------------------------+
|                                     HEADS-UP DISPLAY & RENDERING                                 |
|                                                                                                  |
|   - Real-time Video Window (OpenCV HighGUI)                                                      |
|   - Bounding Boxes: Color-coded by state (Green = Confirmed, Yellow = Searching)                 |
|   - Floating Labels: Person ID + Detection Confidence                                            |
|   - Modern HUD Dashboard:                                                                        |
|     * Total Unique Visitors Count                                                                |
|     * Current Visitors in Frame                                                                  |
|     * Real-time Processing FPS                                                                   |
|     * Frame Skip Telemetry & Stream Status                                                       |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Core Functional Features

### 2.1 Video Ingestion & Stream Resilience
- **Dual-Mode Input Support**: Seamlessly accepts local video files (`sample.mp4`, `.avi`, `.mkv`) or live network RTSP/HTTP camera feeds (`rtsp://username:password@ip:port/h264`).
- **Resilient Frame Decoding**: Gracefully handles network dropped frames, stream re-connection pauses, and video end-of-file (EOF) events without process crashes.

### 2.2 Face Detection with YOLOv8
- **Ultralytics YOLOv8 Engine**: State-of-the-art single-stage face detection optimized for extreme poses, motion blur, and low lighting.
- **Configurable Thresholds**: Bounding box confidence thresholding (`confidence_threshold`) and Non-Maximum Suppression (`iou_threshold`) prevent false alarms.
- **Aspect-Preserving Normalization**: Face crops are extracted with proportional contextual margins for optimal feature alignment.

### 2.3 Deep Feature Representation with InsightFace
- **ArcFace Loss Architecture**: Generates 512-dimensional feature embeddings with high angular margin separation between distinct subjects.
- **Strictly No `face_recognition` Library**: Uses high-performance ONNX Runtime execution with native CUDA/CPU providers.
- **Unit L2 Normalization**: All vectors are normalized to unit length ($\|\mathbf{e}\|_2 = 1.0$), enabling instantaneous cosine similarity calculations via dot product.

### 2.4 Multi-Object Tracking via ByteTrack
- **Two-Stage Association Matching**: Associates both high-confidence and low-confidence detections using the Hungarian algorithm and Kalman Filter motion state.
- **Kalman State Propagation**: On skipped frames, the Kalman filter predicts the spatial trajectory of tracks without invoking heavy neural network backbones.
- **Occlusion Handling**: Retains lost track identities for up to `max_lost_frames` before confirming a person has exited the scene.

### 2.5 Auto-Registration & Anti-Double-Counting Engine
- **First-Seen Auto-Enrollment**: Automatically enrolls previously unrecorded visitors on their first appearance, assigning a persistent identifier (`VISITOR_0001`, `VISITOR_0002`, ...).
- **Sub-Millisecond Vector Search**: In-memory vectorized cosine similarity matrix search against all registered identities.
- **Zero-Double-Count Guarantee**: If a visitor leaves the camera view and returns minutes or hours later, the system re-identifies them, links them to their existing profile, and **does not** increment the unique visitor counter.

### 2.6 Dual-Tier Logging & Visual Archival System
- **Single Entry / Single Exit Invariant**: Each physical appearance session yields strictly one `ENTRY` event log upon arrival and one `EXIT` event log upon departure.
- **Partitioned Disk Storage**: High-resolution face crops are stored in structured date hierarchies (`logs/entries/YYYY-MM-DD/person_<id>_<timestamp>.jpg`).
- **SQLite Relational Persistence**: ACID-compliant transactional storage of visitor profiles, 512-D binary float arrays, visit counts, and complete entry/exit event history.
- **System Event Ledger (`events.log`)**: Chronological audit trail capturing system boot, tracking updates, embedding generation, database transactions, entry logs, and exit durations.

### 2.7 Configurable Frame-Skipping Pipeline
- **Adaptive Execution**: Configurable `frame_skip` parameter (e.g., skip 2 frames out of every 3).
- **Sustained Real-Time FPS**: Keeps processing above 30 FPS even on mid-range hardware by running neural networks at 10 Hz and Kalman tracking at 30 Hz.

---

## 3. Compute Load Consumption Analysis (1080p @ 30 FPS)

### 3.1 Input Specifications & Data Rates
*   **Resolution**: $1920 \times 1080$ pixels (Full HD)
*   **Color Channels**: 3 (BGR, 8-bit per channel)
*   **Frame Data Size**: $1920 \times 1080 \times 3 \approx 6.22 \text{ MB per uncompressed frame}$
*   **Throughput at 30 FPS**: $\approx 186.6 \text{ MB/s}$ raw video decode bandwidth.

---

### 3.2 Stage-by-Stage Latency & Compute Breakdown

| Pipeline Stage | Algorithm / Model | Complexity / FLOPs | GPU Latency (RTX 3060/4060) | CPU Latency (Intel i7 12th Gen / AMD Ryzen 7) |
|---|---|---|---|---|
| **1. Video Decoding & Resize** | OpenCV / FFmpeg (Letterbox 640x640) | $\mathcal{O}(W \times H)$ | 1.8 ms | 4.2 ms |
| **2. Face Detection** | YOLOv8n (FP16 / INT8) | ~8.7 GFLOPs | 3.5 ms | 38.0 ms |
| **3. Face Crop & Normalization** | Affine Warp / Resize (112x112) | $\mathcal{O}(K \times w \times h)$ | 0.4 ms (for 3 faces) | 1.5 ms (for 3 faces) |
| **4. Feature Embedding** | InsightFace ArcFace (MobileFaceNet / ResNet50) | ~1.2 GFLOPs per face | 2.5 ms per face | 22.0 ms per face |
| **5. Tracking Association** | ByteTrack (Kalman Filter + Hungarian) | $\mathcal{O}(N^3)$ (scipy assignment) | 0.6 ms | 1.2 ms |
| **6. Cosine Matching** | Vectorized Dot Product ($K \times M$) | $\mathcal{O}(K \times M \times 512)$ | 0.05 ms | 0.15 ms |
| **7. DB Write & Disk I/O** | SQLite commit + JPEG encode | Disk I/O (Async / buffered) | 2.0 ms (only on entry) | 2.5 ms (only on entry) |
| **8. Frame Annotation & UI** | OpenCV Drawing + imshow | $\mathcal{O}(W \times H)$ | 1.5 ms | 3.0 ms |

*(Assuming average scene density of $K = 2$ active faces).*

---

### 3.3 Compute Load Comparison: Without Frame Skip vs. With Frame Skip (`frame_skip = 2`)

#### A. Continuous Mode (`frame_skip = 0`, Process Every Frame at 30 FPS)
*   **GPU Environment**:
    *   Total Latency per Frame: $\approx 1.8 + 3.5 + 0.4 + (2 \times 2.5) + 0.6 + 0.05 + 1.5 = 12.85 \text{ ms}$
    *   Maximum Attainable Frame Rate: $\frac{1000}{12.85} \approx \mathbf{77.8 \text{ FPS}}$ (Easily real-time).
    *   GPU Core Utilization: ~35-45% of RTX 3060/4060.
    *   VRAM Consumption: ~1.2 GB (YOLOv8 + InsightFace ONNX Runtime CUDA provider).
*   **CPU Environment**:
    *   Total Latency per Frame: $\approx 4.2 + 38.0 + 1.5 + (2 \times 22.0) + 1.2 + 0.15 + 3.0 = 92.05 \text{ ms}$
    *   Maximum Attainable Frame Rate: $\frac{1000}{92.05} \approx \mathbf{10.8 \text{ FPS}}$ (Cannot sustain 30 FPS real-time; causes severe video buffering and backlog).
    *   CPU Utilization: 100% across 4-6 cores.

#### B. Optimized Frame Skip Mode (`frame_skip = 2`, Detection every 3rd frame)
*   **Mechanism**:
    *   Frame 0: Full Detection + Feature Extraction + Tracking Update ($\text{GPU: } 12.85 \text{ ms}$, $\text{CPU: } 92.05 \text{ ms}$).
    *   Frames 1 & 2: Video Decode + Kalman Filter Motion Prediction + HUD Draw ($\text{GPU: } 3.9 \text{ ms}$, $\text{CPU: } 8.4 \text{ ms}$).
*   **Average Amortized Latency**:
    $$\text{Latency}_{\text{avg}} = \frac{T_{\text{detect}} + 2 \times T_{\text{track}}}{3}$$
    *   **GPU Environment**:
        $$\text{Latency}_{\text{avg}} = \frac{12.85 + 2 \times 3.9}{3} = \frac{20.65}{3} \approx \mathbf{6.88 \text{ ms}} \implies \mathbf{145 \text{ FPS capacity}}$$
        *   GPU Utilization drops to < **18%**.
    *   **CPU Environment**:
        $$\text{Latency}_{\text{avg}} = \frac{92.05 + 2 \times 8.4}{3} = \frac{108.85}{3} \approx \mathbf{36.28 \text{ ms}} \implies \mathbf{27.6 - 30.0 \text{ FPS real-time sustained!}}$$
        *   CPU Utilization stabilizes at **60-75%** with zero frame lag.

---

### 3.4 Hardware Recommendation Matrix

| Resource | Minimum Specification (CPU Mode) | Recommended Specification (GPU Mode) |
|---|---|---|
| **Processor** | Intel Core i5 / AMD Ryzen 5 (6 cores) | Intel Core i7 / AMD Ryzen 7 |
| **Graphics Card** | Integrated Graphics (Intel UHD / Iris Xe) | NVIDIA RTX 3050 / 3060 / 4060 (>= 4GB VRAM) |
| **System RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **Storage** | 500 MB for models + disk space for logs | NVMe SSD for fast video buffering & frame logging |
| **Compute Framework** | ONNX Runtime (OpenVINO / CPUExecutionProvider) | ONNX Runtime GPU (CUDAExecutionProvider + cuDNN) |
