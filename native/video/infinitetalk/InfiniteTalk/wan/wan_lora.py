import os
import torch
from safetensors import safe_open
from loguru import logger
import gc
from functools import lru_cache
from tqdm import tqdm

@lru_cache(maxsize=None)
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE")
    return RUNNING_FLAG

class WanLoraWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        # self.override_dict = {}  # On CPU

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        # Ensure lora_name is not empty
        if not lora_name:
            lora_name = f"lora_{len(self.lora_metadata)}"

        if lora_name in self.lora_metadata:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        # Validate lora_path exists
        if not os.path.exists(lora_path):
            logger.error(f"LoRA file not found: {lora_path}")
            return None

        self.lora_metadata[lora_name] = {"path": lora_path}
        logger.info(f"Registered LoRA metadata for: '{lora_name}' from {lora_path}")

        return lora_name

    def _load_lora_file(self, file_path, param_dtype):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(param_dtype) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0, param_dtype=torch.bfloat16, device='cpu'):
        if lora_name not in self.lora_metadata:
            logger.error(f"LoRA '{lora_name}' not found. Please load it first.")
            return False

        try:
            lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"], param_dtype)
            # weight_dict = self.model.original_weight_dict
            self._apply_lora_weights(lora_weights, alpha, device)
            # self.model._init_weights(weight_dict)

            logger.info(f"Applied LoRA: '{lora_name}' with alpha={alpha}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply LoRA '{lora_name}': {str(e)}")
            return False

    def get_parameter_by_name(self, model, param_name):
        parts = param_name.split('.')
        current = model
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    @torch.no_grad()
    def _apply_lora_weights(self, lora_weights, alpha, device):
        lora_pairs = {}
        prefix = "diffusion_model."

        # Validate device parameter
        if isinstance(device, str):
            if device == 'cpu':
                target_device = torch.device('cpu')
            elif device.startswith('cuda'):
                target_device = torch.device(device)
            else:
                logger.warning(f"Invalid device string '{device}', defaulting to CPU")
                target_device = torch.device('cpu')
        else:
            target_device = device

        for key in lora_weights.keys():
            if key.endswith("lora_down.weight") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("lora_down.weight", "weight")
                b_key = key.replace("lora_down.weight", "lora_up.weight")
                if b_key in lora_weights:
                    lora_pairs[base_name] = (key, b_key)
            elif key.endswith("diff_b") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("diff_b", "bias")
                lora_pairs[base_name] = (key)
            elif key.endswith("diff") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("diff", "weight")
                lora_pairs[base_name] = (key)

        applied_count = 0
        for name in tqdm(lora_pairs.keys(), desc="Loading LoRA weights"):
            try:
                param = self.get_parameter_by_name(self.model, name)
                if str(target_device) == 'cpu':
                    dtype = torch.float32
                else:
                    dtype = param.dtype
                    
                if isinstance(lora_pairs[name], tuple):
                    name_lora_A, name_lora_B = lora_pairs[name]
                    lora_A = lora_weights[name_lora_A].to(target_device, dtype)
                    lora_B = lora_weights[name_lora_B].to(target_device, dtype)
                    delta = torch.matmul(lora_B, lora_A) * alpha
                    delta = delta.to(param.device, param.dtype)
                    param.add_(delta)
                else:
                    name_lora = lora_pairs[name]
                    delta = lora_weights[name_lora].to(param.device, dtype) * alpha
                    delta = delta.to(param.device, param.dtype)
                    param.add_(delta)
                applied_count += 1
            except Exception as e:
                logger.warning(f"Failed to apply LoRA weight for '{name}': {e}")
                continue

        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
            )


    def list_loaded_loras(self):
        return list(self.lora_metadata.keys())

    def get_current_lora(self):
        return self.model.current_lora