from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from enum import Enum
import fal_client
import tempfile
import os
import logging

class GeometryFormatEnum(str, Enum):
    """Output geometry file format options."""
    glb = "glb"
    usdz = "usdz"
    fbx = "fbx"
    obj = "obj"
    stl = "stl"

class MaterialEnum(str, Enum):
    """Material type options."""
    pbr = "PBR"
    shaded = "Shaded"
    all = "All"

class QualityMeshEnum(str, Enum):
    """Mesh quality and resolution options."""
    triangle_500k = "500K Triangle"
    triangle_200k = "200K Triangle"
    triangle_100k = "100K Triangle"
    quad_4k = "4K Quad"
    quad_2k = "2K Quad"
    quad_1k = "1K Quad"

class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(
        None,
        description="Text guidance for 3D model generation (e.g., 'A futuristic robot with sleek metallic design')"
    )
    input_images: Optional[List[File]] = Field(
        None,
        description="Up to 5 reference images for 3D generation (optional if prompt is provided)"
    )
    geometry_file_format: GeometryFormatEnum = Field(
        GeometryFormatEnum.glb,
        description="Output format for the 3D model geometry"
    )
    material: MaterialEnum = Field(
        MaterialEnum.all,
        description="Material type for the 3D model"
    )
    quality_mesh_option: QualityMeshEnum = Field(
        QualityMeshEnum.triangle_500k,
        description="Mesh resolution and type"
    )
    use_original_alpha: bool = Field(
        False,
        description="Use original alpha channel from input images"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible generation"
    )
    preview_render: bool = Field(
        False,
        description="Generate preview renders of the 3D model"
    )

class AppOutput(BaseAppOutput):
    model_mesh: File = Field(description="Generated 3D model file")
    textures: List[File] = Field(description="Generated texture files")
    seed: int = Field(description="Seed used for generation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/hyper3d/rodin/v2"

        self.logger.info("Rodin 3D Generator initialized successfully")

    def _upload_file_to_url(self, file_path: str) -> str:
        """Upload a local file to temporary storage for processing."""
        try:
            file_url = fal_client.upload_file(file_path)
            self.logger.info(f"File uploaded to temporary storage successfully")
            return file_url
        except Exception as e:
            self.logger.error(f"Failed to upload file {file_path}: {e}")
            raise RuntimeError(f"Failed to upload file: {e}")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        request_data = {
            "geometry_file_format": input_data.geometry_file_format.value,
            "material": input_data.material.value,
            "quality_mesh_option": input_data.quality_mesh_option.value,
            "use_original_alpha": input_data.use_original_alpha,
            "preview_render": input_data.preview_render,
        }

        # Add prompt if provided
        if input_data.prompt:
            request_data["prompt"] = input_data.prompt

        # Add seed if provided
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        # Upload and add input images if provided
        if input_data.input_images:
            image_urls = []
            for img_file in input_data.input_images[:5]:  # Max 5 images
                image_url = self._upload_file_to_url(img_file.path)
                image_urls.append(image_url)
            request_data["input_image_urls"] = image_urls

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate 3D model using Rodin."""
        try:
            # Validate input - need either prompt or images
            if not input_data.prompt and not input_data.input_images:
                raise RuntimeError("Either prompt or input images must be provided")

            # Validate input files if provided
            if input_data.input_images:
                for img in input_data.input_images:
                    if not img.exists():
                        raise RuntimeError(f"Input image does not exist at path: {img.path}")

            # Set up API key from environment
            api_key = os.environ.get("API_KEY")
            if not api_key:
                raise RuntimeError(
                    "API_KEY environment variable is required for model access."
                )

            fal_client.api_key = api_key

            self.logger.info("Starting 3D model generation...")
            if input_data.prompt:
                self.logger.info(f"Prompt: {input_data.prompt}")
            if input_data.input_images:
                self.logger.info(f"Using {len(input_data.input_images)} reference images")
            self.logger.info(f"Output format: {input_data.geometry_file_format.value}")
            self.logger.info(f"Mesh quality: {input_data.quality_mesh_option.value}")

            # Prepare request data for model
            request_data = self._prepare_model_request(input_data)

            self.logger.info("Initializing model inference...")

            # Define progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"Model: {log['message']}")

            # Run model inference with progress logging
            result = fal_client.subscribe(
                self.model_id,
                arguments=request_data,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            self.logger.info("3D model generation completed successfully")

            # Process the generated model mesh
            model_url = result["model_mesh"]["url"]
            self.logger.info("Processing generated 3D model...")

            # Determine output file extension
            ext = f".{input_data.geometry_file_format.value}"

            # Create temporary file for the model
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                model_path = tmp_file.name

            # Download model content
            import requests
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Process texture files
            texture_files = []
            for idx, texture_data in enumerate(result.get("textures", [])):
                texture_url = texture_data["url"]

                # Determine texture file extension
                texture_ext = ".png"  # Default to PNG
                if "content_type" in texture_data:
                    if "jpeg" in texture_data["content_type"] or "jpg" in texture_data["content_type"]:
                        texture_ext = ".jpg"

                with tempfile.NamedTemporaryFile(suffix=texture_ext, delete=False) as tmp_tex:
                    texture_path = tmp_tex.name

                tex_response = requests.get(texture_url, stream=True)
                tex_response.raise_for_status()

                with open(texture_path, "wb") as f:
                    for chunk in tex_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                texture_files.append(File(path=texture_path))

            self.logger.info(f"3D model and {len(texture_files)} textures processed successfully")

            # Prepare output
            return AppOutput(
                model_mesh=File(path=model_path),
                textures=texture_files,
                seed=result.get("seed", input_data.seed or 0)
            )

        except Exception as e:
            self.logger.error(f"Error during 3D generation: {e}")
            raise RuntimeError(f"3D generation failed: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        self.logger.info("Rodin 3D Generator unloaded successfully")
