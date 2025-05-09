import os
from huggingface_hub import hf_hub_download

def download_model():
    print("Downloading LTX-Video model...")
    model_dir = os.environ.get("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    hf_hub_download(
        repo_id="Lightricks/LTX-Video", 
        filename="ltx-video-2b-v0.9.5.safetensors", 
        local_dir=model_dir, 
        local_dir_use_symlinks=False, 
        repo_type='model'
    )
    print(f"Model downloaded to {model_dir}")

if __name__ == "__main__":
    download_model() 