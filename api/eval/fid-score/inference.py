import logging
import numpy as np
from typing import List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from scipy import linalg


class RunInput(BaseAppInput):
    reference_images: List[File] = Field(description="Reference/real image set")
    generated_images: List[File] = Field(description="Generated/test image set")


class RunOutput(BaseAppOutput):
    fid_score: float = Field(description="FID score (lower = more similar distributions)")
    reference_count: int = Field(description="Number of reference images processed")
    generated_count: int = Field(description="Number of generated images processed")


class App(BaseApp):
    async def setup(self, config):
        import torch
        from torchvision import transforms
        from torchvision.models import inception_v3

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading InceptionV3 for FID computation...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.logger.info(f"InceptionV3 loaded on {self.device}")

    def _get_activations(self, image_files: List[File]) -> np.ndarray:
        import torch
        from PIL import Image

        activations = []
        for img_file in image_files:
            img = Image.open(img_file.path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                act = self.model(tensor)
            activations.append(act.cpu().numpy().flatten())
        return np.array(activations)

    def _compute_fid(self, act1: np.ndarray, act2: np.ndarray) -> float:
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    async def run(self, input_data: RunInput) -> RunOutput:
        self.logger.info(f"Computing FID: {len(input_data.reference_images)} ref vs {len(input_data.generated_images)} gen images")

        ref_act = self._get_activations(input_data.reference_images)
        gen_act = self._get_activations(input_data.generated_images)

        fid = self._compute_fid(ref_act, gen_act)

        self.logger.info(f"FID score: {fid:.4f}")

        return RunOutput(
            fid_score=fid,
            reference_count=len(input_data.reference_images),
            generated_count=len(input_data.generated_images),
        )
