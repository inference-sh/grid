import logging
from typing import List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field


class RunInput(BaseAppInput):
    prompt: str = Field(description="Text prompt to score alignment against")
    images: List[File] = Field(description="Images to score (1+)")


class RunOutput(BaseAppOutput):
    scores: List[float] = Field(description="PickScore for each image (higher = better alignment)")
    best_index: int = Field(description="Index of the highest-scoring image")


class App(BaseApp):
    async def setup(self, config):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading PickScore v1 model...")
        model_id = "yuvalkirstain/PickScore_v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.logger.info(f"PickScore loaded on {self.device}")

    async def run(self, input_data: RunInput) -> RunOutput:
        import torch
        from PIL import Image

        self.logger.info(f"Scoring {len(input_data.images)} image(s) against prompt: {input_data.prompt[:80]}")

        pil_images = [Image.open(img.path).convert("RGB") for img in input_data.images]

        image_inputs = self.processor(
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=input_data.prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs_out = self.model.get_image_features(**image_inputs)
            image_embs = image_embs_out if isinstance(image_embs_out, torch.Tensor) else image_embs_out.pooler_output
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs_out = self.model.get_text_features(**text_inputs)
            text_embs = text_embs_out if isinstance(text_embs_out, torch.Tensor) else text_embs_out.pooler_output
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)

        score_list = scores[0].cpu().tolist()
        best_idx = scores[0].argmax().item()

        self.logger.info(f"Scores: {score_list}, best index: {best_idx}")

        return RunOutput(
            scores=score_list,
            best_index=best_idx,
        )
