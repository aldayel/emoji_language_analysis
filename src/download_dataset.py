import os
import kagglehub
from pathlib import Path

def download_dataset():
    """Download the Tweets With Emoji dataset and save it to the data/raw directory."""
    # Create data directories if they don't exist
    data_dir = Path(__file__).parent.parent / 'data'
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("ericwang1011/tweets-with-emoji", path=str(raw_dir))
    print(f"Dataset downloaded to: {path}")

if __name__ == "__main__":
    download_dataset()
