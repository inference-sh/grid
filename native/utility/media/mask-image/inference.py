from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from PIL import Image
import os
import tempfile


class AppInput(BaseAppInput):
    input_image: File = Field(description="The input image to be masked")
    mask_image: File = Field(description="The mask image (can be PNG with transparency, or black/white image)")
    invert_mask: bool = Field(default=False, description="Whether to invert/reverse the mask before applying it")


class AppOutput(BaseAppOutput):
    masked_image: File = Field(description="The resulting partially transparent masked image")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize resources if needed."""
        pass

    def _extract_mask_alpha(self, mask_img):
        """
        Extract or create an alpha channel from various mask image formats.
        
        Args:
            mask_img: PIL Image in any format
            
        Returns:
            PIL Image in 'L' mode representing the alpha channel
        """
        if mask_img.mode == "RGBA":
            # Extract existing alpha channel
            return mask_img.split()[-1]
        elif mask_img.mode == "RGB":
            # Convert RGB to grayscale for mask
            return mask_img.convert("L")
        elif mask_img.mode == "L":
            # Already grayscale, use as-is
            return mask_img
        elif mask_img.mode == "1":
            # Black and white mode - convert to grayscale
            return mask_img.convert("L")
        elif mask_img.mode == "P":
            # Palette mode - convert to RGBA first, then extract alpha if present
            rgba_mask = mask_img.convert("RGBA")
            if rgba_mask.mode == "RGBA":
                return rgba_mask.split()[-1]
            else:
                return rgba_mask.convert("L")
        else:
            # For any other mode, convert to grayscale
            return mask_img.convert("L")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Apply the semi-transparent mask to the input image."""
        
        # Check if input files exist
        if not input_data.input_image.exists():
            raise RuntimeError(f"Input image does not exist at path: {input_data.input_image.path}")
        
        if not input_data.mask_image.exists():
            raise RuntimeError(f"Mask image does not exist at path: {input_data.mask_image.path}")
        
        try:
            # Load the input image and convert to RGBA for transparency support
            input_img = Image.open(input_data.input_image.path).convert("RGBA")
            
            # Load the mask image
            mask_img = Image.open(input_data.mask_image.path)
            
            # Resize mask to match input image dimensions if they differ
            if mask_img.size != input_img.size:
                mask_img = mask_img.resize(input_img.size, Image.Resampling.LANCZOS)
            
            # Extract or create alpha channel from mask
            mask_alpha = self._extract_mask_alpha(mask_img)
            
            # Invert mask if requested
            if input_data.invert_mask:
                mask_alpha = Image.eval(mask_alpha, lambda x: 255 - x)
            
            # Create the masked image using the mask's alpha channel
            # This combines the input image with the mask's transparency
            masked_img = Image.composite(
                input_img,  # Foreground (input image)
                Image.new("RGBA", input_img.size, (0, 0, 0, 0)),  # Background (transparent)
                mask_alpha  # Mask alpha channel
            )
            
            # Save the result to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Save as PNG to preserve transparency
            masked_img.save(output_path, "PNG")
            
            return AppOutput(
                masked_image=File(path=output_path)
            )
            
        except Exception as e:
            raise RuntimeError(f"Error processing images: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        pass