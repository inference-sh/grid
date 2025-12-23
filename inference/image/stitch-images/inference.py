from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Literal
from PIL import Image
import os

class AppInput(BaseAppInput):
    image1: File = Field(description="The first image to stitch")
    image2: File = Field(description="The second image to stitch")
    orientation: Literal["horizontal", "vertical", "auto"] = Field(description="Direction to stitch images: 'horizontal' (side by side), 'vertical' (top to bottom), or 'auto' (automatically choose direction that maximizes final image size)")
    squared_output: bool = Field(default=False, description="If True, create a squared output image with white padding instead of scaling down images")
    max_width: int = Field(default=1024, description="Maximum width of the final output image")
    max_height: int = Field(default=1024, description="Maximum height of the final output image")

class AppOutput(BaseAppOutput):
    stitched_image: File = Field(description="The stitched image file")

# For LLM apps, you can use the LLMInput and LLMInputWithImage classes for convenience
# from inferencesh import LLMInput, LLMInputWithImage
# The LLMInput class provides a standard structure for LLM-based applications with:
# - system_prompt: Sets the AI assistant's role and behavior
# - context: List of previous conversation messages between user and assistant
# - text: The current user's input prompt
#
# Example usage:
# class AppInput(LLMInput):
#     additional_field: str = Field(description="Any additional input needed")

# The LLMInputWithImage class extends LLMInput to support image inputs by adding:
# - image: Optional File field for providing images to vision-capable models
#
# Example usage:
# class AppInput(LLMInputWithImage):
#     additional_field: str = Field(description="Any additional input needed")

# Each ContextMessage in the context list contains:
# - role: Either "user", "assistant", or "system"
# - text: The message content
#
# ContextMessageWithImage adds:
# - image: Optional File field for messages containing images



class App(BaseApp):
    def _determine_optimal_orientation(self, width1, height1, width2, height2, squared_output, max_width=1024, max_height=1024):
        """
        Determine the optimal orientation that balances image quality preservation for both images.
        
        Returns:
            str: Either "horizontal" or "vertical"
        """
        if squared_output:
            # For rectangular output, calculate how well each orientation fits in the target canvas
            # Use similar area calculation as for regular mode since we need to consider target dimensions
            area1 = width1 * height1
            area2 = width2 * height2
            target_area = (area1 * area2) ** 0.5
            
            aspect1 = width1 / height1
            aspect2 = width2 / height2
            
            target_height1 = int((target_area / aspect1) ** 0.5)
            target_width1 = int(target_height1 * aspect1)
            target_height2 = int((target_area / aspect2) ** 0.5)
            target_width2 = int(target_height2 * aspect2)
            
            # Calculate how much space each orientation would need
            horizontal_width = target_width1 + target_width2
            horizontal_height = max(target_height1, target_height2)
            
            vertical_width = max(target_width1, target_width2)
            vertical_height = target_height1 + target_height2
            
            # Calculate efficiency (how much of the target canvas is used)
            # Higher efficiency is better
            horizontal_efficiency = (horizontal_width * horizontal_height) / (max_width * max_height)
            vertical_efficiency = (vertical_width * vertical_height) / (max_width * max_height)
            
            return "horizontal" if horizontal_efficiency >= vertical_efficiency else "vertical"
            
        else:
            # For non-squared output, use a balanced scoring system that considers both images
            
            # Calculate scaling factors for horizontal orientation
            target_height_horizontal = min(height1, height2)
            scale1_horizontal = target_height_horizontal / height1
            scale2_horizontal = target_height_horizontal / height2
            
            # Calculate scaling factors for vertical orientation  
            target_width_vertical = min(width1, width2)
            scale1_vertical = target_width_vertical / width1
            scale2_vertical = target_width_vertical / width2
            
            # Calculate quality preservation scores (higher is better)
            # Score considers: final area + balance between images + penalty for heavy downscaling
            
            # Horizontal scoring
            area1_horizontal = width1 * height1 * (scale1_horizontal ** 2)
            area2_horizontal = width2 * height2 * (scale2_horizontal ** 2)
            total_area_horizontal = area1_horizontal + area2_horizontal
            
            # Balance factor: penalize if one image is scaled much more than the other
            scale_ratio_horizontal = max(scale1_horizontal, scale2_horizontal) / min(scale1_horizontal, scale2_horizontal)
            balance_penalty_horizontal = 1.0 / (1.0 + (scale_ratio_horizontal - 1.0) * 0.5)
            
            # Downscaling penalty: penalize heavy downscaling
            avg_scale_horizontal = (scale1_horizontal + scale2_horizontal) / 2
            downscale_penalty_horizontal = min(avg_scale_horizontal, 1.0) ** 0.5
            
            horizontal_score = total_area_horizontal * balance_penalty_horizontal * downscale_penalty_horizontal
            
            # Vertical scoring
            area1_vertical = width1 * height1 * (scale1_vertical ** 2)
            area2_vertical = width2 * height2 * (scale2_vertical ** 2)
            total_area_vertical = area1_vertical + area2_vertical
            
            scale_ratio_vertical = max(scale1_vertical, scale2_vertical) / min(scale1_vertical, scale2_vertical)
            balance_penalty_vertical = 1.0 / (1.0 + (scale_ratio_vertical - 1.0) * 0.5)
            
            avg_scale_vertical = (scale1_vertical + scale2_vertical) / 2
            downscale_penalty_vertical = min(avg_scale_vertical, 1.0) ** 0.5
            
            vertical_score = total_area_vertical * balance_penalty_vertical * downscale_penalty_vertical
            
            # Choose orientation with higher balanced score
            return "horizontal" if horizontal_score >= vertical_score else "vertical"

    async def setup(self, metadata):
        """Initialize resources for image processing."""
        # No special setup needed for PIL image processing
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Stitch two images together horizontally or vertically with optional squared output and final resizing."""
        
        # Check if input files exist and are accessible
        if not input_data.image1.exists():
            raise RuntimeError(f"First image does not exist at path: {input_data.image1.path}")
        if not input_data.image2.exists():
            raise RuntimeError(f"Second image does not exist at path: {input_data.image2.path}")
        
        # Load the two images
        try:
            img1 = Image.open(input_data.image1.path)
            img2 = Image.open(input_data.image2.path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
        except Exception as e:
            raise RuntimeError(f"Failed to load images: {str(e)}")
        
        # Get original dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Determine actual orientation to use
        if input_data.orientation == "auto":
            actual_orientation = self._determine_optimal_orientation(width1, height1, width2, height2, input_data.squared_output, input_data.max_width, input_data.max_height)
        else:
            actual_orientation = input_data.orientation
        
        # Calculate similar dimensions for both images to create balanced appearance
        # Use the geometric mean of areas to find a balanced target size
        area1 = width1 * height1
        area2 = width2 * height2
        target_area = (area1 * area2) ** 0.5  # Geometric mean of areas
        
        # Calculate aspect ratios
        aspect1 = width1 / height1
        aspect2 = width2 / height2
        
        # Resize both images to similar areas while preserving aspect ratios
        target_height1 = int((target_area / aspect1) ** 0.5)
        target_width1 = int(target_height1 * aspect1)
        
        target_height2 = int((target_area / aspect2) ** 0.5)
        target_width2 = int(target_height2 * aspect2)
        
        # Resize images to similar dimensions
        img1_resized = img1.resize((target_width1, target_height1), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((target_width2, target_height2), Image.Resampling.LANCZOS)
        
        if input_data.squared_output:
            # For squared output, place resized images and pad to max_width x max_height
            
            if actual_orientation == "horizontal":
                # Place images side by side
                total_width = target_width1 + target_width2
                total_height = max(target_height1, target_height2)
                
                # Create canvas
                stitched_image = Image.new('RGB', (total_width, total_height), color='white')
                
                # Calculate vertical positions to center images
                y1 = (total_height - target_height1) // 2
                y2 = (total_height - target_height2) // 2
                
                # Paste images
                stitched_image.paste(img1_resized, (0, y1))
                stitched_image.paste(img2_resized, (target_width1, y2))
                
            else:  # vertical
                # Place images top to bottom
                total_width = max(target_width1, target_width2)
                total_height = target_height1 + target_height2
                
                # Create canvas
                stitched_image = Image.new('RGB', (total_width, total_height), color='white')
                
                # Calculate horizontal positions to center images
                x1 = (total_width - target_width1) // 2
                x2 = (total_width - target_width2) // 2
                
                # Paste images
                stitched_image.paste(img1_resized, (x1, 0))
                stitched_image.paste(img2_resized, (x2, target_height1))
            
            # Pad to the specified max_width x max_height dimensions with white
            target_canvas_width = input_data.max_width
            target_canvas_height = input_data.max_height
            
            # Create the target canvas
            padded_image = Image.new('RGB', (target_canvas_width, target_canvas_height), color='white')
            
            # Calculate scaling to fit the stitched image within the target canvas
            scale_x = target_canvas_width / total_width
            scale_y = target_canvas_height / total_height
            scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            if scale_factor < 1.0:
                # Scale down the stitched image to fit
                scaled_width = int(total_width * scale_factor)
                scaled_height = int(total_height * scale_factor)
                stitched_image_scaled = stitched_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                stitched_image.close()
            else:
                # Use original size
                scaled_width = total_width
                scaled_height = total_height
                stitched_image_scaled = stitched_image
            
            # Center the stitched image in the target canvas
            paste_x = (target_canvas_width - scaled_width) // 2
            paste_y = (target_canvas_height - scaled_height) // 2
            padded_image.paste(stitched_image_scaled, (paste_x, paste_y))
            
            final_image = padded_image
            stitched_image_scaled.close()
            
        else:
            # For regular output, further align dimensions based on orientation
            if actual_orientation == "horizontal":
                # For horizontal stitching, make heights exactly equal
                target_height = min(target_height1, target_height2)
                
                # Recalculate widths for exact height match
                final_width1 = int(target_width1 * (target_height / target_height1))
                final_width2 = int(target_width2 * (target_height / target_height2))
                
                # Final resize for exact alignment
                img1_final = img1_resized.resize((final_width1, target_height), Image.Resampling.LANCZOS)
                img2_final = img2_resized.resize((final_width2, target_height), Image.Resampling.LANCZOS)
                
                # Create new image with combined width
                total_width = final_width1 + final_width2
                final_image = Image.new('RGB', (total_width, target_height))
                
                # Paste images side by side
                final_image.paste(img1_final, (0, 0))
                final_image.paste(img2_final, (final_width1, 0))
                
                img1_final.close()
                img2_final.close()
                
            else:  # vertical
                # For vertical stitching, make widths exactly equal
                target_width = min(target_width1, target_width2)
                
                # Recalculate heights for exact width match
                final_height1 = int(target_height1 * (target_width / target_width1))
                final_height2 = int(target_height2 * (target_width / target_width2))
                
                # Final resize for exact alignment
                img1_final = img1_resized.resize((target_width, final_height1), Image.Resampling.LANCZOS)
                img2_final = img2_resized.resize((target_width, final_height2), Image.Resampling.LANCZOS)
                
                # Create new image with combined height
                total_height = final_height1 + final_height2
                final_image = Image.new('RGB', (target_width, total_height))
                
                # Paste images top to bottom
                final_image.paste(img1_final, (0, 0))
                final_image.paste(img2_final, (0, final_height1))
                
                img1_final.close()
                img2_final.close()
        
        # Clean up resized images
        img1_resized.close()
        img2_resized.close()
        
        # Final resize to fit within max dimensions
        current_width, current_height = final_image.size
        
        # Calculate scale factor to fit within max dimensions
        width_scale = input_data.max_width / current_width
        height_scale = input_data.max_height / current_height
        scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale
        
        if scale_factor < 1.0:
            new_width = int(current_width * scale_factor)
            new_height = int(current_height * scale_factor)
            resized_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            final_image.close()
            final_image = resized_image
        
        # Save the final image
        output_path = "/tmp/stitched_image.jpg"
        final_image.save(output_path, format='JPEG', quality=95)
        
        # Clean up PIL images from memory
        img1.close()
        img2.close()
        final_image.close()
        
        return AppOutput(
            stitched_image=File(path=output_path)
        )

    async def unload(self):
        """Clean up resources here."""
        pass