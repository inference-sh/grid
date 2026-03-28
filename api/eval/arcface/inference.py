import logging
import numpy as np
from typing import List, Optional
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import BaseModel, Field
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis


class FaceInfo(BaseModel):
    confidence: float = Field(description="Detection confidence (0-1)")
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")


class RunInput(BaseAppInput):
    source: File = Field(description="Source/reference face image")
    target: File = Field(description="Target image to compare identity against")


class BatchInput(BaseAppInput):
    source: File = Field(description="Source/reference face image (identity anchor)")
    targets: List[File] = Field(description="Target images to compare against the source")


class EmbedInput(BaseAppInput):
    image: File = Field(description="Face image to extract embedding from")


class CompareInput(BaseAppInput):
    source_embedding: List[float] = Field(description="Source face embedding (512-d vector)")
    target_embedding: List[float] = Field(description="Target face embedding (512-d vector)")


class RunOutput(BaseAppOutput):
    similarity: float = Field(description="Cosine similarity between face embeddings (-1 to 1)")
    source_face_detected: bool = Field(description="Whether a face was detected in the source")
    target_face_detected: bool = Field(description="Whether a face was detected in the target")
    source_face: Optional[FaceInfo] = Field(None, description="Source face detection info")
    target_face: Optional[FaceInfo] = Field(None, description="Target face detection info")


class BatchOutput(BaseAppOutput):
    similarities: List[float] = Field(description="Cosine similarity for each target image")
    source_face_detected: bool = Field(description="Whether a face was detected in the source")
    target_faces_detected: List[bool] = Field(description="Whether a face was detected in each target")
    mean_similarity: float = Field(description="Mean similarity across all targets with detected faces")
    source_face: Optional[FaceInfo] = Field(None, description="Source face detection info")
    target_faces: List[Optional[FaceInfo]] = Field(description="Detection info for each target")


class EmbedOutput(BaseAppOutput):
    embedding: Optional[List[float]] = Field(description="512-d normalized face embedding, null if no face detected")
    face_detected: bool = Field(description="Whether a face was detected")
    face: Optional[FaceInfo] = Field(None, description="Face detection info")


class CompareOutput(BaseAppOutput):
    similarity: float = Field(description="Cosine similarity between the two embeddings (-1 to 1)")


class App(BaseApp):
    async def setup(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading ArcFace model via InsightFace...")
        import onnxruntime
        available = onnxruntime.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        self.logger.info(f"ONNX providers: {providers}")
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.logger.info("ArcFace ready")

    def _get_face(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _face_info(self, face) -> FaceInfo:
        return FaceInfo(
            confidence=float(face.det_score),
            bbox=[float(x) for x in face.bbox],
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    async def run(self, input_data: RunInput) -> RunOutput:
        self.logger.info("Computing identity similarity between source and target")

        src_face = self._get_face(input_data.source.path)
        tgt_face = self._get_face(input_data.target.path)

        src_detected = src_face is not None
        tgt_detected = tgt_face is not None

        similarity = 0.0
        if src_detected and tgt_detected:
            similarity = self._cosine_similarity(src_face.normed_embedding, tgt_face.normed_embedding)

        self.logger.info(f"Similarity: {similarity:.4f} (src_face={src_detected}, tgt_face={tgt_detected})")

        return RunOutput(
            similarity=similarity,
            source_face_detected=src_detected,
            target_face_detected=tgt_detected,
            source_face=self._face_info(src_face) if src_detected else None,
            target_face=self._face_info(tgt_face) if tgt_detected else None,
        )

    async def batch(self, input_data: BatchInput) -> BatchOutput:
        self.logger.info(f"Batch: comparing source against {len(input_data.targets)} targets")

        src_face = self._get_face(input_data.source.path)
        src_detected = src_face is not None

        similarities = []
        tgt_detected = []
        tgt_faces_info = []

        for target in input_data.targets:
            tgt_face = self._get_face(target.path)
            detected = tgt_face is not None
            tgt_detected.append(detected)
            tgt_faces_info.append(self._face_info(tgt_face) if detected else None)

            if src_detected and detected:
                sim = self._cosine_similarity(src_face.normed_embedding, tgt_face.normed_embedding)
            else:
                sim = 0.0
            similarities.append(sim)

        valid_sims = [s for s, d in zip(similarities, tgt_detected) if d and src_detected]
        mean_sim = float(np.mean(valid_sims)) if valid_sims else 0.0

        self.logger.info(f"Batch done: mean_similarity={mean_sim:.4f}, {sum(tgt_detected)}/{len(tgt_detected)} faces detected")

        return BatchOutput(
            similarities=similarities,
            source_face_detected=src_detected,
            target_faces_detected=tgt_detected,
            mean_similarity=mean_sim,
            source_face=self._face_info(src_face) if src_detected else None,
            target_faces=tgt_faces_info,
        )

    async def embed(self, input_data: EmbedInput) -> EmbedOutput:
        self.logger.info("Extracting face embedding")

        face = self._get_face(input_data.image.path)
        detected = face is not None

        self.logger.info(f"Face detected: {detected}")

        return EmbedOutput(
            embedding=face.normed_embedding.tolist() if detected else None,
            face_detected=detected,
            face=self._face_info(face) if detected else None,
        )

    async def compare_embeddings(self, input_data: CompareInput) -> CompareOutput:
        src = np.array(input_data.source_embedding)
        tgt = np.array(input_data.target_embedding)

        similarity = self._cosine_similarity(src, tgt)

        self.logger.info(f"Embedding comparison: similarity={similarity:.4f}")

        return CompareOutput(similarity=similarity)
