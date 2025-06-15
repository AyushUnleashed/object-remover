import os
import sys
import tempfile
import torch
from typing import Optional
from cog import BasePredictor, Input, Path
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

# Add minimax_remover to Python path so its internal imports work
minimax_remover_path = os.path.join(os.path.dirname(__file__), "minimax_remover")
if minimax_remover_path not in sys.path:
    sys.path.insert(0, minimax_remover_path)

# Now import the MiniMax-Remover components
from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline

# Import our utility functions and download function
from utils import (
    get_video_info, validate_inputs, load_video_from_path, load_mask_from_path,
    calculate_safe_resolution, calculate_output_fps, MAX_FPS, MAX_HEIGHT, MAX_WIDTH
)
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
                scheduler=self.scheduler
            ).to(device)
            
            self.device = device
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(
        self,
        video: Path = Input(
            description="Input video file with objects to be removed"
        ),
        mask: Path = Input(
            description="Mask video file where white areas indicate objects to remove. See examples: https://replicate.com/ayushunleashed/minimax-remover/readme"
        ),
        num_frames: int = Input(
            description="Number of frames to process (-1 = same as original video)",
            default=-1,
            ge=-1
        ),
        height: int = Input(
            description=f"Output video height (-1 = same as original video, auto-scaled to max {MAX_HEIGHT}p if needed)",
            default=-1,
            ge=-1
        ),
        width: int = Input(
            description=f"Output video width (-1 = same as original video, auto-scaled to max {MAX_WIDTH}px if needed)", 
            default=-1,
            ge=-1
        ),
        fps: int = Input(
            description=f"Output video FPS (-1 = same as original video, max {MAX_FPS} fps)",
            default=-1,
            ge=-1,
            le=MAX_FPS
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (higher = better quality, slower. 6=fast, 8=balanced, 12=high quality)",
            default=8,
            ge=1,
            le=50
        ),
        mask_dilation_iterations: int = Input(
            description="Mask expansion iterations for robust removal (higher = more thorough removal)",
            default=8,
            ge=1,
            le=20
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results (leave blank for random)",
            default=None
        ),
    ) -> Path:
        """Run video object removal with smart defaults"""
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        # Validate inputs and get video info
        print("Validating input videos...")
        original_info, mask_info = validate_inputs(str(video), str(mask))
        
        # Determine actual parameters based on defaults and inputs
        actual_frames = original_info['total_frames'] if num_frames == -1 else min(num_frames, original_info['total_frames'])
        
        # Determine resolution
        if height == -1 or width == -1:
            actual_height, actual_width = calculate_safe_resolution(
                original_info['height'], 
                original_info['width']
            )
        else:
            actual_height, actual_width = calculate_safe_resolution(height, width)
        
        # Determine FPS
        output_fps = calculate_output_fps(original_info['fps'], fps)
        
        print(f"📹 Processing {actual_frames} frames at {actual_width}x{actual_height}")
        print(f"🎬 Output FPS: {output_fps} (original: {original_info['fps']:.1f})")
        print(f"⚙️ Quality: {num_inference_steps} inference steps")
        print(f"🎯 Mask dilation: {mask_dilation_iterations} iterations")
        
        # Load video and mask with determined frame count
        print("Loading original video and mask...")
        video_frames = load_video_from_path(str(video), actual_frames)
        mask_frames = load_mask_from_path(str(mask), actual_frames)
        
        print(f"Video shape: {video_frames.shape}")
        print(f"Mask shape: {mask_frames.shape}")
        
        # Ensure both videos use the same number of frames (take minimum)
        min_frames = min(video_frames.shape[0], mask_frames.shape[0])
        if min_frames != actual_frames:
            print(f"Adjusting to {min_frames} frames based on available video content")
        
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
                height=actual_height,
                width=actual_width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                iterations=mask_dilation_iterations
            ).frames[0]
            
            print("Inference completed successfully!")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                raise RuntimeError(
                    f"GPU memory error. Try reducing: num_frames (current: {min_frames}), "
                    f"resolution (current: {actual_width}x{actual_height}), or num_inference_steps (current: {num_inference_steps})"
                )
            raise e
        
        # Save output video with calculated FPS
        output_path = Path(tempfile.mkdtemp()) / "output.mp4"
        export_to_video(result, str(output_path), fps=output_fps)
        
        print(f"✅ Video saved to: {output_path}")
        print(f"📊 Output: {len(result)} frames at {output_fps} FPS")
        return output_path