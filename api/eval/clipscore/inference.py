import logging
from typing import List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field


class RunInput(BaseAppInput):
    prompt: str = Field(description="Text prompt to measure adherence against")
    images: List[File] = Field(description="Images to score (1+)")


class RunOutput(BaseAppOutput):
    scores: List[float] = Field(description="CLIP cosine similarity for each image (0-1, higher = better adherence)")
    best_index: int = Field(description="Index of the best-matching image")


class App(BaseApp):
    async def setup(self, config):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading CLIP ViT-L/14...")
        model_id = "openai/clip-vit-large-patch14"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).eval().to(self.device)
        self.logger.info(f"CLIP loaded on {self.device}")

    async def run(self, input_data: RunInput) -> RunOutput:
        import torch
        from PIL import Image

        self.logger.info(f"Scoring {len(input_data.images)} image(s) against prompt: {input_data.prompt[:80]}")

        pil_images = [Image.open(img.path).convert("RGB") for img in input_data.images]

        inputs = self.processor(
            text=[input_data.prompt],
            images=pil_images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image is [num_images, num_texts]
            # Normalize to cosine similarity (0-1 range via sigmoid-ish, but raw cosine is more interpretable)
            image_embs = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embs = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores = (image_embs @ text_embs.T).squeeze(-1)

        score_list = scores.cpu().tolist()
        best_idx = scores.argmax().item()

        self.logger.info(f"Scores: {score_list}, best index: {best_idx}")

        return RunOutput(scores=score_list, best_index=best_idx)
