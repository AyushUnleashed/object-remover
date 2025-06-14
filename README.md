# MiniMax-Remover Cog Wrapper

This repository contains a Cog wrapper for the MiniMax-Remover video object removal model, deployed on Replicate at `ayushunleashed/minimax-remover`.

## Overview

MiniMax-Remover is a fast and effective video object remover based on minimax optimization. This Cog wrapper provides a convenient API for running the model on Replicate with video and mask inputs.

## Repository Structure

```
object-remover/
├── cog.yaml                    # Cog configuration
├── predict.py                  # Cog prediction interface
├── download_weights.py         # Weight downloader script
├── minimax_remover/           # Git submodule
│   ├── README.md
│   ├── requirements.txt
│   ├── pipeline_minimax_remover.py
│   ├── transformer_minimax_remover.py
│   └── ...
├── sample_data/               # Sample videos for testing
│   ├── racoon_video.mp4       # Input video with racoon
│   └── racoon_mask.mp4        # Mask video (white areas to remove)
└── README.md                  # This file
```

## Local Testing with Cog

### 1. Clone with Submodule

```bash
# Clone the repository
git clone https://github.com/AyushUnleashed/object-remover.git
cd object-remover

# Initialize and update the submodule
git submodule update --init --recursive
```

### 2. Install Cog

Follow the [official Cog installation guide](https://cog.run/getting-started/):

```bash
# On macOS
brew install replicate/tap/cog

# On Linux/Windows WSL
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

### 3. Test Locally

```bash
# Build the Docker image (weights download automatically during setup)
cog build

# Or manually download weights first (optional)
python download_weights.py

# Test with sample videos
cog predict -i video=@sample_data/racoon_video.mp4 -i mask=@sample_data/racoon_mask.mp4
```

## Usage

### Input Requirements

- **Video**: MP4 format recommended, max 81 frames
- **Mask**: Video file where white areas indicate objects to remove
- Both video and mask should have the same frame count

### API Parameters

- `video` (required): Input video file
- `mask` (required): Mask video file  
- `num_frames`: Number of frames to process (1-81, default: 25)
- `height`: Output video height (256-1024, default: 480)
- `width`: Output video width (256-1024, default: 832)
- `num_inference_steps`: Denoising steps (1-50, default: 12)
- `iterations`: Mask dilation iterations (1-20, default: 6)
- `seed`: Random seed (optional)

### Example Usage

#### Local Testing with Cog

```bash
# Basic usage with sample data
cog predict \
    -i video=@sample_data/racoon_video.mp4 \
    -i mask=@sample_data/racoon_mask.mp4

# With custom parameters
cog predict \
    -i video=@sample_data/racoon_video.mp4 \
    -i mask=@sample_data/racoon_mask.mp4 \
    -i num_frames=30 \
    -i height=512 \
    -i width=768 \
    -i num_inference_steps=8 \
    -i iterations=4 \
    -i seed=42
```

#### Python API (Using Deployed Model)

```python
import replicate

output = replicate.run(
    "ayushunleashed/minimax-remover",
    input={
        "video": open("your_video.mp4", "rb"),
        "mask": open("your_mask.mp4", "rb"),
        "num_frames": 25,
        "height": 480,
        "width": 832,
        "num_inference_steps": 12,
        "iterations": 6
    }
)

print(f"Output video: {output}")
```

## Model Details

- **Architecture**: Simplified DiT (Diffusion Transformer) with minimax optimization
- **Inference Steps**: 6-12 steps (much faster than traditional diffusion models)
- **Memory Requirements**: ~8GB GPU memory for typical usage
- **Supported Resolutions**: Up to 1024x1024 pixels
- **Model Weights**: Downloaded automatically from Hugging Face during first setup

## Performance Tips

1. **Frame Count**: Fewer frames = faster processing
2. **Resolution**: Lower resolution = faster processing  
3. **Inference Steps**: 6-12 steps provide good quality/speed balance
4. **Mask Quality**: Clean masks with clear boundaries work best

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_frames`, `height`, or `width`
2. **Slow Performance**: Reduce `num_inference_steps` to 6-8
3. **Poor Quality**: Increase `num_inference_steps` or improve mask quality

### Debug Mode

Enable verbose logging by setting environment variable:

```bash
export COG_LOG_LEVEL=debug
cog predict -i video=@sample_data/racoon_video.mp4 -i mask=@sample_data/racoon_mask.mp4
```

## License

This Cog wrapper follows the same license as the original MiniMax-Remover project. See the [original repository](https://github.com/zibojia/MiniMax-Remover) for license details.

## Citation

If you use this model, please cite the original MiniMax-Remover paper:

```bibtex
@article{minimax2024,
  title={MiniMax-Remover: Taming Bad Noise Helps Video Object Removal},
  author={Bojia Zi and Weixuan Peng and Xianbiao Qi and Jianan Wang and Shihao Zhao and Rong Xiao and Kam-Fai Wong},
  year={2024}
}
```

## Support

- [Cog Documentation](https://cog.run/)
- [Replicate Documentation](https://replicate.com/docs)