from huggingface_hub import snapshot_download
import os

def download_model():
    repo_id = "tencent/SongGeneration"

    return snapshot_download(
        repo_id=repo_id,
        local_dir=".",
        revision="0c80d30",
        token=os.environ.get("HF_TOKEN"), 
        ignore_patterns=['.git*']
    )

if __name__ == '__main__':
    download_model()
