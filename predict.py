import os
import sys
import tempfile
import torch
from typing import Optional
from cog import BasePredictor, Input, Path
import numpy as np
from decord import VideoReader
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

# Add minimax_remover to Python path so its internal imports work
minimax_remover_path = os.path.join(os.path.dirname(__file__), "minimax_remover")
if minimax_remover_path not in sys.path:
    sys.path.insert(0, minimax_remover_path)


# Import the MiniMax-Remover components (these will be from the submodule)
from minimax_remover.pipeline_minimax_remover import Minimax_Remover_Pipeline
from minimax_remover.transformer_minimax_remover import Transformer3DModel

# Import our download function
from download_weights import download_weights, verify_downloads

MODEL_CACHE = "./model_weights"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading MiniMax-Remover model...")
        
        # Download weights if not present
        if not verify_downloads():
            print("Downloading model weights...")
            download_weights()
        
        # Load model components
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load individual components
            self.vae = AutoencoderKLWan.from_pretrained(
                os.path.join(MODEL_CACHE, "vae"), 
                torch_dtype=torch.float16
            )
            
            self.transformer = Transformer3DModel.from_pretrained(
                os.path.join(MODEL_CACHE, "transformer"), 
                torch_dtype=torch.float16
            )
            
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                os.path.join(MODEL_CACHE, "scheduler")
            )
            
            # Initialize the pipeline
            self.pipe = Minimax_Remover_Pipeline(
                vae=self.vae,
                transformer=self.transformer,
                scheduler=self.scheduler,
            ).to(device)
            
            self.device = device
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def load_video_from_path(self, video_path: str, max_frames: int = 81) -> torch.Tensor:
        """Load video frames from file path and convert to tensor"""
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            # Take frames evenly spaced if video is longer than max_frames
            if total_frames > max_frames:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                frame_indices = list(range(min(total_frames, max_frames)))
            
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Convert to tensor and normalize to [-1, 1]
            frames_tensor = torch.from_numpy(frames) / 127.5 - 1.0
            
            return frames_tensor
            
        except Exception as e:
            print(f"Error loading video: {e}")
            raise e

    def load_mask_from_path(self, mask_path: str, max_frames: int = 81) -> torch.Tensor:
        """Load mask frames from file path and convert to tensor"""
        try:
            vr = VideoReader(mask_path)
            total_frames = len(vr)
            
            # Take frames evenly spaced if mask video is longer than max_frames
            if total_frames > max_frames:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                frame_indices = list(range(min(total_frames, max_frames)))
            
            masks = vr.get_batch(frame_indices).asnumpy()
            
            # Convert to tensor and process mask
            masks_tensor = torch.from_numpy(masks)
            
            # Take only first channel if RGB, and ensure single channel
            if masks_tensor.shape[-1] == 3:
                masks_tensor = masks_tensor[:, :, :, :1]
            
            # Threshold the mask
            masks_tensor[masks_tensor > 20] = 255
            masks_tensor[masks_tensor < 255] = 0
            masks_tensor = masks_tensor / 255.0
            
            return masks_tensor
            
        except Exception as e:
            print(f"Error loading mask: {e}")
            raise e

    def predict(
        self,
        video: Path = Input(
            description="Input video file (MP4 format recommended)"
        ),
        mask: Path = Input(
            description="Mask video file where white areas indicate objects to remove"
        ),
        num_frames: int = Input(
            description="Number of frames to process (max 81)",
            default=25,
            ge=1,
            le=81
        ),
        height: int = Input(
            description="Output video height",
            default=480,
            ge=256,
            le=1024
        ),
        width: int = Input(
            description="Output video width", 
            default=832,
            ge=256,
            le=1024
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=12,
            ge=1,
            le=50
        ),
        iterations: int = Input(
            description="Mask dilation iterations for robustness",
            default=6,
            ge=1,
            le=20
        ),
        seed: Optional[int] = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
    ) -> Path:
        """Run video object removal"""
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        # Load video and mask
        print("Loading video and mask...")
        video_frames = self.load_video_from_path(str(video), num_frames)
        mask_frames = self.load_mask_from_path(str(mask), num_frames)
        
        print(f"Video shape: {video_frames.shape}")
        print(f"Mask shape: {mask_frames.shape}")
        
        # Ensure both have the same number of frames
        min_frames = min(video_frames.shape[0], mask_frames.shape[0])
        video_frames = video_frames[:min_frames]
        mask_frames = mask_frames[:min_frames]
        
        # Run inference
        print("Running MiniMax-Remover inference...")
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            result = self.pipe(
                images=video_frames,
                masks=mask_frames,
                num_frames=min_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                iterations=iterations
            ).frames[0]
            
            print("Inference completed successfully!")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e
        
        # Save output video
        output_path = Path(tempfile.mkdtemp()) / "output.mp4"
        export_to_video(result, str(output_path), fps=16)
        
        print(f"Video saved to: {output_path}")
        return output_path