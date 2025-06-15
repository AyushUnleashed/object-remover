import numpy as np
import torch
from decord import VideoReader

# Constants - modify here to change limits across the application
MAX_FPS = 30
DEFAULT_FPS = 16
MIN_DIMENSION = 64
MAX_HEIGHT = 1080  # 1080p height limit
MAX_WIDTH = 1920   # Typical 1080p width limit for 16:9


def get_video_info(video_path: str):
    """Get video properties"""
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        height, width = vr[0].shape[:2]
        
        # Try to get FPS, fallback to default if not available
        try:
            fps = vr.get_avg_fps()
        except:
            fps = DEFAULT_FPS
        
        return {
            'total_frames': total_frames,
            'height': height,
            'width': width,
            'fps': fps
        }
    except Exception as e:
        raise ValueError(f"Could not read video file: {e}")


def validate_inputs(original_video_path: str, mask_video_path: str):
    """Validate that original video and mask video are compatible"""
    try:
        original_info = get_video_info(original_video_path)
        mask_info = get_video_info(mask_video_path)
        
        # Check frame count compatibility
        if original_info['total_frames'] != mask_info['total_frames']:
            min_frames = min(original_info['total_frames'], mask_info['total_frames'])
            print(f"Frame count difference: Original video has {original_info['total_frames']} frames, "
                    f"mask video has {mask_info['total_frames']} frames. Will process first {min_frames} frames.")
        
        # Check resolution compatibility (allow some tolerance)
        orig_aspect = original_info['width'] / original_info['height']
        mask_aspect = mask_info['width'] / mask_info['height']
        
        if abs(orig_aspect - mask_aspect) > 0.1:
            print(f"Warning: Aspect ratio mismatch - Original: {orig_aspect:.2f}, Mask: {mask_aspect:.2f}")
        
        return original_info, mask_info
        
    except Exception as e:
        raise ValueError(f"Input validation failed: {e}")


def load_video_from_path(video_path: str, num_frames: int = -1) -> torch.Tensor:
    """Load video frames from file path and convert to tensor"""
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        
        # Determine actual frames to process
        if num_frames == -1:
            actual_frames = total_frames
        else:
            actual_frames = min(num_frames, total_frames)
        
        # Take frames evenly spaced if video is longer than what we want to process
        if total_frames > actual_frames:
            frame_indices = np.linspace(0, total_frames - 1, actual_frames, dtype=int)
        else:
            frame_indices = list(range(actual_frames))
        
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Convert to tensor and normalize to [-1, 1]
        frames_tensor = torch.from_numpy(frames) / 127.5 - 1.0
        
        print(f"Loaded {actual_frames} frames from {total_frames} total frames")
        return frames_tensor
        
    except Exception as e:
        print(f"Error loading video: {e}")
        raise e


def load_mask_from_path(mask_path: str, num_frames: int = -1) -> torch.Tensor:
    """Load mask frames from file path and convert to tensor"""
    try:
        vr = VideoReader(mask_path)
        total_frames = len(vr)
        
        # Determine actual frames to process
        if num_frames == -1:
            actual_frames = total_frames
        else:
            actual_frames = min(num_frames, total_frames)
        
        # Take frames evenly spaced if mask video is longer than what we want to process
        if total_frames > actual_frames:
            frame_indices = np.linspace(0, total_frames - 1, actual_frames, dtype=int)
        else:
            frame_indices = list(range(actual_frames))
        
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
        
        print(f"Loaded {actual_frames} mask frames")
        return masks_tensor
        
    except Exception as e:
        print(f"Error loading mask: {e}")
        raise e


def calculate_safe_resolution(original_height: int, original_width: int):
    """Calculate safe resolution maintaining aspect ratio with 1080p max"""
    height = max(MIN_DIMENSION, original_height)
    width = max(MIN_DIMENSION, original_width)
    
    # Check if resolution exceeds 1080p limits
    if height > MAX_HEIGHT or width > MAX_WIDTH:
        print(f"Original resolution {width}x{height} exceeds 1080p limits, scaling down...")
        
        # Calculate scaling factors for both dimensions
        height_scale = MAX_HEIGHT / height if height > MAX_HEIGHT else 1.0
        width_scale = MAX_WIDTH / width if width > MAX_WIDTH else 1.0
        
        # Use the more restrictive scaling factor to maintain aspect ratio
        scale_factor = min(height_scale, width_scale)
        
        # Apply scaling
        height = int(height * scale_factor)
        width = int(width * scale_factor)
        
        print(f"Scaled to {width}x{height} (scale factor: {scale_factor:.3f})")
    
    # Ensure dimensions are even (required for video encoding)
    height = height - (height % 2)
    width = width - (width % 2)
    
    return height, width


def calculate_output_fps(original_fps: float, requested_fps: int) -> int:
    """Calculate output FPS with constraints"""
    if requested_fps == -1:
        # Use original FPS but clamp to MAX_FPS
        output_fps = min(MAX_FPS, max(1, int(original_fps)))
    else:
        # Use requested FPS but clamp to MAX_FPS
        output_fps = min(MAX_FPS, max(1, requested_fps))
    
    return output_fps