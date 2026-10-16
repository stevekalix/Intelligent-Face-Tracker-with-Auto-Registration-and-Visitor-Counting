"""
Face Processing Module for Intelligent Face Tracker & Visitor Counting.
Integrates Ultralytics YOLOv8 for face detection and InsightFace for 512-D ArcFace
feature embedding extraction and cosine similarity matching.
Strictly avoids the 'face_recognition' library.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("FaceProcessor")


class FaceProcessor:
    """Wrapper coordinating YOLOv8 face detection and InsightFace biometrics."""

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        insightface_model_name: str = "buffalo_s",
        similarity_threshold: float = 0.50,
        device: str = "auto"
    ):
        """Initialize YOLOv8 detector and InsightFace recognition engine.

        Args:
            yolo_model_path: Path or name of YOLOv8 model weights.
            confidence_threshold: Minimum detection confidence score.
            iou_threshold: Non-Maximum Suppression IoU threshold.
            insightface_model_name: InsightFace model pack name (e.g. 'buffalo_s', 'buffalo_l').
            similarity_threshold: Cosine similarity cutoff for identity verification.
            device: 'cuda', 'cpu', or 'auto'.
        """
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Determine execution provider
        self.use_cuda = False
        if device == "auto":
            try:
                import torch
                self.use_cuda = torch.cuda.is_available()
            except ImportError:
                self.use_cuda = False
        elif device.lower() in ("cuda", "gpu"):
            self.use_cuda = True

        logger.info("Initializing FaceProcessor (Device: %s | CUDA Available: %s)",
                    "CUDA" if self.use_cuda else "CPU", self.use_cuda)

        # 1. Initialize YOLOv8 Detector
        self._init_yolo(yolo_model_path)

        # 2. Initialize InsightFace Recognizer
        self._init_insightface(insightface_model_name)

    def _init_yolo(self, model_path: str) -> None:
        """Load YOLOv8 model via Ultralytics."""
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8 detector: %s", model_path)
            self.yolo_model = YOLO(model_path)
            if self.use_cuda:
                self.yolo_model.to("cuda")
            logger.info("YOLOv8 detector loaded successfully.")
        except Exception as e:
            logger.error("Failed to load YOLOv8 model from [%s]: %s", model_path, e)
            raise

    def _init_insightface(self, model_name: str) -> None:
        """Load InsightFace model pack for facial feature embedding extraction."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_cuda else ["CPUExecutionProvider"]
            logger.info("Loading InsightFace app pack [%s] with providers: %s", model_name, providers)

            # Initialize FaceAnalysis app
            self.face_app = FaceAnalysis(
                name=model_name,
                providers=providers,
                allowed_modules=["detection", "recognition"]
            )
            # Prepare with ctx_id (0 for GPU, -1 for CPU)
            ctx_id = 0 if self.use_cuda else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            # Access the underlying ArcFace recognition model
            self.rec_model = None
            if hasattr(self.face_app, "models") and "recognition" in self.face_app.models:
                self.rec_model = self.face_app.models["recognition"]
                logger.info("InsightFace recognition engine (ArcFace 512-D) ready.")
            else:
                logger.warning("InsightFace recognition module not directly isolated; using standard app pipeline.")
        except Exception as e:
            logger.error("Error loading InsightFace: %s", e)
            logger.warning("Ensure 'insightface' and ONNX models are installed properly.")
            self.face_app = None
            self.rec_model = None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[List[int], float, np.ndarray]]:
        """Run YOLOv8 face detection on an input BGR frame.

        Args:
            frame: Input video frame (numpy BGR image).

        Returns:
            List of tuples: (bbox, confidence, face_crop)
            where bbox is [x1, y1, x2, y2] integers.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections = []

        try:
            # Run inference with YOLOv8
            results = self.yolo_model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,
                device=0 if self.use_cuda else "cpu"
            )

            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(confs))

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cls_id = int(classes[i])

                # Clamp to frame dimensions
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y2 = max(y1 + 1, min(int(y2), h))

                # Filter out tiny or invalid boxes
                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue

                # Add a 10% context margin for biometric cropping
                margin_x = int((x2 - x1) * 0.10)
                margin_y = int((y2 - y1) * 0.10)
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w, x2 + margin_x)
                crop_y2 = min(h, y2 + margin_y)

                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                detections.append(([x1, y1, x2, y2], conf, face_crop))

        except Exception as e:
            logger.error("Exception during YOLOv8 detection: %s", e)

        return detections

    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Optional[List[int]] = None,
        face_crop: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Extract a 512-D L2-normalized facial embedding vector using InsightFace.

        Args:
            frame: Full video frame or face crop.
            bbox: [x1, y1, x2, y2] bounding box within frame.
            face_crop: Pre-cropped face image (optional).

        Returns:
            Normalized 512-dimensional float32 numpy array, or None if extraction failed.
        """
        if self.face_app is None:
            logger.warning("InsightFace is not initialized. Cannot extract embeddings.")
            return None

        t0 = time.perf_counter()

        try:
            # Case A: If face_crop is provided, analyze the crop directly
            if face_crop is not None and face_crop.size > 0:
                faces = self.face_app.get(face_crop)
                if faces and len(faces) > 0:
                    # Pick the highest detection score or largest face in crop
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    embedding = best_face.normed_embedding
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.debug("Extracted embedding from crop in %.2f ms", duration_ms)
                    return embedding.astype(np.float32)

            # Case B: If full frame and bbox are provided
            if frame is not None and frame.size > 0:
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    # Crop context area
                    sub_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if sub_img.size > 0:
                        faces = self.face_app.get(sub_img)
                        if faces and len(faces) > 0:
                            embedding = faces[0].normed_embedding
                            return embedding.astype(np.float32)

                # Fallback: Run InsightFace on frame region
                faces = self.face_app.get(frame)
                if faces and len(faces) > 0:
                    if bbox is not None:
                        # Find closest face matching bbox center
                        bx_center = (bbox[0] + bbox[2]) / 2.0
                        by_center = (bbox[1] + bbox[3]) / 2.0
                        best_face = min(
                            faces,
                            key=lambda f: ((f.bbox[0] + f.bbox[2])/2 - bx_center)**2 + ((f.bbox[1] + f.bbox[3])/2 - by_center)**2
                        )
                        return best_face.normed_embedding.astype(np.float32)
                    return faces[0].normed_embedding.astype(np.float32)

        except Exception as e:
            logger.error("Exception in extract_embedding: %s", e)

        return None

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute the cosine similarity between two feature vectors.

        Formula: sim(u, v) = (u . v) / (||u|| * ||v||)

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(dot / (norm1 * norm2))

    def find_matching_visitor(
        self,
        query_embedding: np.ndarray,
        known_embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        """Perform vectorized matching of query embedding against all registered visitors.

        Args:
            query_embedding: 512-D normalized vector.
            known_embeddings: Dict mapping person_id -> 512-D normalized vector.
            threshold: Minimum similarity threshold (defaults to config threshold).

        Returns:
            Tuple of (best_matching_person_id or None, highest_similarity_score).
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if not known_embeddings or query_embedding is None:
            return None, 0.0

        person_ids = list(known_embeddings.keys())
        # Stack all known vectors into matrix M x 512
        matrix = np.vstack([known_embeddings[pid] for pid in person_ids])

        # Query vector shape 512
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-6)

        # Vectorized dot products: (M, 512) @ (512,) -> (M,)
        similarities = np.dot(matrix, q)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            best_person_id = person_ids[best_idx]
            logger.debug("Matched identity [%s] with similarity %.4f (threshold: %.2f)",
                         best_person_id, best_score, threshold)
            return best_person_id, best_score

        logger.debug("No match found. Best candidate [%s] score: %.4f < threshold: %.2f",
                     person_ids[best_idx], best_score, threshold)
        return None, best_score
