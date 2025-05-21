from modal import Image, Stub, method, Secret
from modal_config import get_base_image, GPU_CONFIG, MODEL_CONFIG, REQUIRED_SECRETS, AUDIO_CONFIG
import os
import torch
from orpheus_tts import OrpheusModel
import wave
import io

# Create stub
stub = Stub("orpheus-tts")

# Create image with dependencies
image = get_base_image()

@stub.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[Secret.from_name(s) for s in REQUIRED_SECRETS.values()]
)
class OrpheusTTSDeployment:
    def __enter__(self):
        """Initialize the model during container startup"""
        # Login to HuggingFace
        import huggingface_hub
        huggingface_hub.login(token=os.environ["HUGGING_FACE_TOKEN"])

        # Initialize model
        self.model = OrpheusModel(
            model_name=MODEL_CONFIG["model_name"],
            tokenizer=MODEL_CONFIG["tokenizer"],
            max_model_len=MODEL_CONFIG["max_model_len"]
        )

    @method()
    def generate_speech(self, text: str, voice: str = "tara") -> bytes:
        """Generate speech from text"""
        try:
            # Input validation
            if not text:
                raise ValueError("Text input cannot be empty")

            if voice not in ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]:
                raise ValueError("Invalid voice selection")

            # Generate audio using streaming interface from README example
            buffer = io.BytesIO()

            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(AUDIO_CONFIG["channels"])
                wf.setsampwidth(AUDIO_CONFIG["bits_per_sample"] // 8)
                wf.setframerate(AUDIO_CONFIG["sample_rate"])

                syn_tokens = self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    repetition_penalty=1.1,  # From README examples
                    stop_token_ids=[128258],
                    max_tokens=2000,
                    temperature=0.4,
                    top_p=0.9
                )

                for audio_chunk in syn_tokens:
                    wf.writeframes(audio_chunk)

            return buffer.getvalue()

        except Exception as e:
            raise RuntimeError(f"Speech generation failed: {str(e)}")

    @method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, "model"),
            "gpu_available": torch.cuda.is_available()
        }

# Example usage
@stub.local_entrypoint()
def main():
    deployment = OrpheusTTSDeployment()
    text = "Hello, this is a test of the Orpheus TTS system."
    audio_data = deployment.generate_speech.remote(text)

    # Save test output
    with open("test_output.wav", "wb") as f:
        f.write(audio_data)