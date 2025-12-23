"""
LoRA helper utilities for loading adapters from various sources.
Supports HuggingFace, Civitai, direct URLs, local files, and uploaded File objects.
"""

import logging
import os
import re
from typing import Optional

import requests
from dateutil.parser import parse as parse_date
from pydantic import BaseModel, Field, model_validator

from inferencesh import File


class LoraConfig(BaseModel):
    """Configuration for a single LoRA adapter.
    
    Provide either lora_url OR lora_file (not both).
    """
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_url: Optional[str] = Field(
        default=None,
        description="URL to LoRA file (.safetensors), Civitai model page, or HuggingFace repo"
    )
    lora_file: Optional[File] = Field(
        default=None,
        description="Uploaded LoRA file (.safetensors)"
    )
    lora_multiplier: float = Field(
        default=1.0,
        description="Multiplier for the LoRA effect (0.0-2.0 typical)"
    )

    @model_validator(mode='after')
    def validate_lora_source(self) -> 'LoraConfig':
        """Ensure at least one of lora_url or lora_file is provided."""
        if not self.lora_url and not self.lora_file:
            raise ValueError("Either 'lora_url' or 'lora_file' must be provided")
        return self
    
    def get_lora_path(self) -> str | None:
        """Get the resolved local path for the LoRA file.
        
        Returns the file path if lora_file is provided, otherwise None.
        """
        if self.lora_file:
            return self.lora_file.path
        return None
    
    def get_lora_identifier(self) -> str:
        """Get a unique identifier for this LoRA (path or URL)."""
        if self.lora_file:
            return self.lora_file.path
        return self.lora_url or ""


def get_civit_download_url(model_id: str, base_model_filter: list[str] | None = None) -> str | None:
    """
    Fetch the download URL for a Civitai model.
    
    Args:
        model_id: The Civitai model ID
        base_model_filter: Optional list of base model names to filter by (e.g., ['Z-Image', 'Z-Image Turbo'])
    
    Returns:
        Download URL or None if not found
    """
    url = f"https://civitai.com/api/v1/models/{model_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        versions = data.get("modelVersions", [])
        
        def get_date(item):
            date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
            try:
                return parse_date(date_str)
            except Exception:
                return parse_date('1970-01-01T00:00:00Z')
        
        sorted_data = sorted(versions, key=get_date, reverse=True)
        
        # Filter by base model if specified
        if base_model_filter:
            filtered_versions = [v for v in sorted_data if v.get('baseModel') in base_model_filter]
            if filtered_versions:
                latest_version = filtered_versions[0]
            elif sorted_data:
                latest_version = sorted_data[0]
            else:
                return None
        elif sorted_data:
            latest_version = sorted_data[0]
        else:
            return None
        
        return latest_version.get("downloadUrl")
    except Exception as e:
        logging.error(f"Failed to fetch Civitai model info: {e}")
        return None


def download_model_data(model_id: str, download_url: str, target_folder: str) -> str:
    """
    Download a model file from a URL to a local folder.
    
    Args:
        model_id: Identifier for the model (used for filename)
        download_url: URL to download from
        target_folder: Local folder to save the file
    
    Returns:
        Path to the downloaded file
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    file_path = os.path.join(target_folder, f"{model_id}.safetensors")
    if not os.path.isfile(file_path):
        # If downloading from civitai, add token if present
        civitai_token = os.environ.get('CIVITAI_TOKEN')
        if 'civitai.com' in download_url and civitai_token:
            if '?' in download_url:
                download_url = f"{download_url}&token={civitai_token}"
            else:
                download_url = f"{download_url}?token={civitai_token}"
        response = requests.get(download_url)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
    return file_path


def load_lora_from_path(pipeline, lora_path: str, adapter_name: str) -> bool:
    """Load a LoRA adapter from a local file path."""
    try:
        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        logging.info(f"Loaded LoRA adapter '{adapter_name}' from file: {lora_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to load LoRA adapter '{adapter_name}' from {lora_path}: {e}")
        return False


def load_lora_adapter(
    pipeline,
    lora_url: str,
    adapter_name: str = "lora",
    lora_multiplier: float = 1.0,
    base_model_filter: list[str] | None = None,
) -> bool:
    """
    Load a LoRA adapter using the modern Diffusers PEFT integration.
    Supports Hugging Face repos, direct URLs, Civitai models, and local files.
    
    Args:
        pipeline: Diffusers pipeline with LoRA support
        lora_url: URL or path to the LoRA weights
        adapter_name: Name to assign to this adapter
        lora_multiplier: Weight multiplier (not used during load, applied via set_adapters)
        base_model_filter: Optional list of base model names for Civitai filtering
    
    Returns:
        True if successfully loaded, False otherwise
    """
    if not lora_url:
        return False
    
    try:
        # 1. Hugging Face blob URL
        if "huggingface.co" in lora_url and "/blob/" in lora_url:
            parts = lora_url.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
                repo_start = parts.index('huggingface.co') + 1
                blob_index = parts.index('blob')
                repo_id = '/'.join(parts[repo_start:blob_index])
                weight_name = '/'.join(parts[blob_index + 2:])
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
            else:
                logging.error(f"Invalid Hugging Face blob URL format: {lora_url}")
                return False
        
        # 2. Hugging Face resolve URL
        elif "huggingface.co" in lora_url and "/resolve/" in lora_url:
            parts = lora_url.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'resolve' in parts:
                repo_start = parts.index('huggingface.co') + 1
                resolve_index = parts.index('resolve')
                repo_id = '/'.join(parts[repo_start:resolve_index])
                weight_name = '/'.join(parts[resolve_index + 2:])
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
            else:
                logging.error(f"Invalid Hugging Face resolve URL format: {lora_url}")
                return False
        
        # 3. Hugging Face repository string (e.g., "username/repo" or "username/repo/filename.safetensors")
        elif "/" in lora_url and not lora_url.startswith('http') and "civitai.com" not in lora_url and not os.path.isfile(lora_url):
            parts = lora_url.split('/')
            if len(parts) == 2:
                repo_id = lora_url
                pipeline.load_lora_weights(repo_id, adapter_name=adapter_name)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}")
                return True
            elif len(parts) > 2:
                repo_id = '/'.join(parts[:2])
                weight_name = '/'.join(parts[2:])
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
        
        # 4. Direct .safetensors URL
        elif lora_url.endswith('.safetensors') and lora_url.startswith('http'):
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
            if not os.path.exists(lora_dir):
                os.makedirs(lora_dir)
            
            model_id = os.path.splitext(os.path.basename(lora_url))[0]
            lora_path = download_model_data(model_id, lora_url, lora_dir)
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            logging.info(f"Loaded LoRA adapter '{adapter_name}' from URL: {lora_url}")
            return True
        
        # 5. Civitai model URL
        elif "civitai.com" in lora_url:
            match = re.search(r"/models/(\d+)", lora_url)
            if not match:
                logging.error("Could not extract model ID from Civitai URL")
                return False
            
            model_id = match.group(1)
            download_url = get_civit_download_url(model_id, base_model_filter)
            if not download_url:
                logging.error(f"No download URL found for Civitai model {model_id}")
                return False
            
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
            if not os.path.exists(lora_dir):
                os.makedirs(lora_dir)
            
            lora_path = download_model_data(model_id, download_url, lora_dir)
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            logging.info(f"Loaded LoRA adapter '{adapter_name}' from Civitai model: {model_id}")
            return True
        
        # 6. Local file path
        elif os.path.isfile(lora_url):
            return load_lora_from_path(pipeline, lora_url, adapter_name)
        
        else:
            logging.warning(f"Unsupported LoRA URL format: {lora_url}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to load LoRA adapter '{adapter_name}': {e}")
        return False


def manage_loras(
    pipeline,
    loras: list[LoraConfig],
    loaded_loras: dict[str, tuple[str, float]],
    base_model_filter: list[str] | None = None,
) -> list[str]:
    """
    Manage LoRA adapters - unload changed ones and load new ones.
    
    Args:
        pipeline: Diffusers pipeline with LoRA support
        loras: List of LoraConfig objects to apply
        loaded_loras: Dict tracking currently loaded adapters {name: (identifier, multiplier)}
                      This dict is modified in-place.
        base_model_filter: Optional list of base model names for Civitai filtering
    
    Returns:
        List of active adapter names
    """
    current_adapters = set(loaded_loras.keys())
    
    # Unload adapters that are no longer requested or have changed
    for adapter_name in list(current_adapters):
        found = next((lora for lora in loras if lora.adapter_name == adapter_name), None)
        if not found:
            # Adapter no longer requested
            try:
                pipeline.delete_adapters(adapter_name)
                logging.info(f"Unloaded previous LoRA adapter: {adapter_name}")
            except Exception as e:
                logging.warning(f"Failed to unload previous LoRA adapter {adapter_name}: {e}")
            del loaded_loras[adapter_name]
        elif (
            loaded_loras[adapter_name][0] != found.get_lora_identifier() or
            loaded_loras[adapter_name][1] != found.lora_multiplier
        ):
            # Adapter changed
            try:
                pipeline.delete_adapters(adapter_name)
                logging.info(f"Unloaded changed LoRA adapter: {adapter_name}")
            except Exception as e:
                logging.warning(f"Failed to unload changed LoRA adapter {adapter_name}: {e}")
            del loaded_loras[adapter_name]
    
    # Load and activate requested adapters
    adapter_weights = []
    active_adapters = []
    for lora in loras:
        lora_identifier = lora.get_lora_identifier()
        
        if (
            lora.adapter_name not in loaded_loras or
            loaded_loras[lora.adapter_name][0] != lora_identifier or
            loaded_loras[lora.adapter_name][1] != lora.lora_multiplier
        ):
            # Load the adapter
            success = False
            lora_path = lora.get_lora_path()
            
            if lora_path:
                # Direct file upload - use the resolved path
                success = load_lora_from_path(pipeline, lora_path, lora.adapter_name)
            elif lora.lora_url:
                # URL-based loading
                success = load_lora_adapter(
                    pipeline, lora.lora_url, lora.adapter_name, lora.lora_multiplier, base_model_filter
                )
            
            if success:
                loaded_loras[lora.adapter_name] = (lora_identifier, lora.lora_multiplier)
        
        if lora.adapter_name in loaded_loras:
            active_adapters.append(lora.adapter_name)
            adapter_weights.append(lora.lora_multiplier)
    
    # Set all active adapters with their weights
    if active_adapters:
        pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
        logging.info(f"Active LoRA adapters: {active_adapters} with weights: {adapter_weights}")
    elif loaded_loras:
        # If no adapters requested but some were loaded, disable them
        try:
            pipeline.set_adapters([])
        except Exception:
            pass  # Some pipelines may not support empty adapter list
    
    return active_adapters
