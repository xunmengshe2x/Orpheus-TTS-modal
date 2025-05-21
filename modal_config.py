from modal import Image, Secret, gpu

# Base image configuration
def get_base_image():
    return (
        Image.debian_slim()
        .apt_install([
            "git",
            "python3-pip",
            "ffmpeg",  # Required for audio processing
        ])
        .pip_install([
            "orpheus-speech",  # Main package
            "vllm==0.7.3",     # Specific version required per README
            "snac",            # Required by orpheus-speech
            "torch",
            "torchaudio",
            "transformers",
            "flash_attn",      # Required for model attention
            "scipy",           # For audio processing
            "numpy",
            "huggingface_hub"
        ])
    )

# GPU Configuration
GPU_CONFIG = gpu.A10G()  # Based on repository requirements for running 3B parameter model

# Model configuration
MODEL_CONFIG = {
    "model_name": "canopylabs/orpheus-tts-0.1-finetune-prod",  # From README
    "tokenizer": "canopylabs/orpheus-3b-0.1-pretrained",
    "max_model_len": 2048,  # From example in README
}

# Environment variables and secrets
REQUIRED_SECRETS = {
    "HF_TOKEN": "HUGGING_FACE_TOKEN",  # For downloading model weights
}

# Audio configuration
AUDIO_CONFIG = {
    "sample_rate": 24000,  # From repository examples
    "channels": 1,
    "bits_per_sample": 16,
}