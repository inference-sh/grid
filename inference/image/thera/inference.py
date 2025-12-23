from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
import pickle
import numpy as np
import jax
from PIL import Image
from huggingface_hub import hf_hub_download
from .model import build_thera
from .super_resolve import process

import sys
print(f"Python version: {sys.version}")

class AppInput(BaseAppInput):
    image: File = Field(description="Input image file")  # Input image file
    scale: float = Field(default=3.92, description="Default scaling factor")  # Default scaling factor
    model: str = Field(default="edsr", description="Model choice: 'edsr' or 'rdn'")  # Model choice: "edsr" or "rdn"
    do_ensemble: bool = Field(default=False, description="Whether to use ensemble")  # Whether to use ensemble

class AppOutput(BaseAppOutput):
    result: File  # Output super-resolved image file

class App(BaseApp):

    
    async def setup(self, metadata):
        """Initialize Thera models."""
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX device type: {jax.devices()[0].device_kind}")

        self.REPO_ID_EDSR = "prs-eth/thera-edsr-pro"
        self.REPO_ID_RDN = "prs-eth/thera-rdn-pro"
        # Load EDSR model
        model_path = hf_hub_download(repo_id=self.REPO_ID_EDSR, filename="model.pkl")
        with open(model_path, 'rb') as fh:
            check = pickle.load(fh)
            self.params_edsr, backbone, size = check['model'], check['backbone'], check['size']
            self.model_edsr = build_thera(3, backbone, size)

        # Load RDN model
        model_path = hf_hub_download(repo_id=self.REPO_ID_RDN, filename="model.pkl")
        with open(model_path, 'rb') as fh:
            check = pickle.load(fh)
            self.params_rdn, backbone, size = check['model'], check['backbone'], check['size']
            self.model_rdn = build_thera(3, backbone, size)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run super-resolution on the input image."""
        # Read input image
        image_in = Image.open(input_data.image.path).convert("RGB")
        source = np.asarray(image_in) / 255.

        # Determine target shape
        target_shape = (
            round(source.shape[0] * input_data.scale),
            round(source.shape[1] * input_data.scale),
        )

        # Select model and parameters
        if input_data.model == 'edsr':
            m, p = self.model_edsr, self.params_edsr
        elif input_data.model == 'rdn':
            m, p = self.model_rdn, self.params_rdn
        else:
            raise ValueError(f"Unknown model: {input_data.model}")

        # Process image
        out = process(source, m, p, target_shape, do_ensemble=input_data.do_ensemble)
        out_image = Image.fromarray(np.asarray(out))

        # Save result
        output_path = "/tmp/thera_result.png"
        out_image.save(output_path)
        
        return AppOutput(result=File(path=output_path))

