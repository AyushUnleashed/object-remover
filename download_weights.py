"""
Download MiniMax-Remover model weights from Hugging Face Hub
"""

import os
import sys
from huggingface_hub import snapshot_download

MODEL_REPO = "zibojia/minimax-remover"
MODEL_CACHE = "./model_weights"

def download_weights():
    """Download model weights from Hugging Face Hub"""
    
    print(f"Downloading MiniMax-Remover weights from {MODEL_REPO}...")
    
    # Create cache directory
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    try:
        # Download the entire model repository
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_CACHE,
            local_dir_use_symlinks=False
        )
        
        print("Model weights downloaded successfully!")
        
        # Verify downloads
        if verify_downloads():
            print("All weights verified successfully!")
        else:
            print("Warning: Some weights may be missing")
            
    except Exception as e:
        print(f"Error downloading weights: {e}")
        sys.exit(1)

def verify_downloads():
    """Verify that all required model files are present"""
    
    required_components = ["vae", "transformer", "scheduler"]
    
    for component in required_components:
        component_path = os.path.join(MODEL_CACHE, component)
        
        if not os.path.exists(component_path):
            return False
            
        # Check for config files
        config_file = os.path.join(component_path, "config.json")
        if not os.path.exists(config_file):
            return False
    
    return True

def check_disk_space():
    """Check available disk space before downloading"""
    
    # Get available disk space
    statvfs = os.statvfs('.')
    available_space_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)

    required_space_gb = 5.0 
    
    if available_space_gb < required_space_gb:
        print(f"Warning: Low disk space. Available: {available_space_gb:.1f}GB, Required: {required_space_gb:.1f}GB")
        return False
    
    print(f"Disk space check passed. Available: {available_space_gb:.1f}GB")
    return True

if __name__ == "__main__":
    print("MiniMax-Remover Weight Downloader")
    print("=" * 40)
    
    # Check disk space
    if not check_disk_space():
        print("Warning: Proceeding with limited disk space...")
    
    # Download weights
    download_weights()
    
    print("Weight download completed!")