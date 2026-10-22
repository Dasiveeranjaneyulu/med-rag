
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from huggingface_hub import snapshot_download
from pathlib import Path

# Create models directory if it doesn't exist
project_root = Path(__file__).parent
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

print("Downloading all-MiniLM-L6-v2...")
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=str(models_dir / "all-MiniLM-L6-v2")
)
print("Done with all-MiniLM-L6-v2!")

print("\nDownloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir=str(models_dir / "ms-marco-MiniLM-L-6-v2")
)
print("Done with cross-encoder!")
