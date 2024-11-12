import os
import huggingface_hub
from huggingface_hub import hf_hub_download
from typing import Optional

class HuggingFaceDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically list model download directories
        base_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_dirs = [d for d in os.listdir(base_models_path) 
                      if os.path.isdir(os.path.join(base_models_path, d))]
        
        return {
            "required": {
                "repo_id": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
                "download_directory": (model_dirs, {}),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional HuggingFace Read Token"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_model"
    CATEGORY = "Model Utilities"
    OUTPUT_NODE = False

    def download_model(self, repo_id: str, filename: str, download_directory: str, 
                       hf_token: Optional[str] = None):
        try:
            # Construct full download path
            base_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            full_download_path = os.path.join(base_models_path, download_directory)
            
            # Prepare token
            token = hf_token.strip() if hf_token else None
            
            # Download the file
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                token=token,
                cache_dir=full_download_path
            )
            
            return (downloaded_file_path,)
        
        except Exception as e:
            print(f"Error downloading model: {e}")
            return ("Download failed",)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "HuggingFaceDownloader": HuggingFaceDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceDownloader": "HuggingFace Model Downloader"
}
