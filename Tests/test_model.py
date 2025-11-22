import onnxruntime as ort
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


class OnnxDF:
    """
    Wrapper for a DeepFilterNet ONNX model.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        # 1. Load the ONNX runtime session
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and ort.get_device() == 'GPU':
            providers.insert(0, "CUDAExecutionProvider")
        
        self.sess = ort.InferenceSession(model_path, providers=providers)
        print(f"Successfully loaded model from {model_path}")
        print(f"Using provider: {self.sess.get_providers()[0]}")

        # 2. Get model input and output names and shapes from the model metadata.
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        
        print("\nModel Inputs:")
        for i in self.sess.get_inputs():
            print(f"- Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

        print("\nModel Outputs:")
        for o in self.sess.get_outputs():
            print(f"- Name: {o.name}, Shape: {o.shape}, Type: {o.type}")

        # Based on torchDF, we can find the model properties
        self.hop_size = 480
        self.fft_size = 960
        self.frame_size = self.hop_size
        
        # Initialize states
        self.states = self.init_states()

    def init_states(self):
        """Initializes the recurrent states of the model."""
        states = []
        for i in self.sess.get_inputs():
            if i.name == "input_frame":
                continue
            # Create a zero tensor with the correct shape for each state
            shape = [d if isinstance(d, int) else 1 for d in i.shape]
            states.append(np.zeros(shape, dtype=np.float32))
        return states

    def __call__(self, audio: torch.Tensor, sr: int):
        """
        Denoises a single-channel audio tensor.

        Args:
            audio (torch.Tensor): A single-channel audio tensor of shape [T,].
            sr (int): The sample rate of the audio.

        Returns:
            torch.Tensor: The denoised audio tensor.
        """
        if sr != 48000:
            raise ValueError("Only 48kHz sample rate is supported.")
        if audio.ndim > 1 and audio.shape[0] > 1:
            print("Warning: Audio has multiple channels, converting to mono.")
            audio = audio.mean(dim=0)

        # Reset states for each new audio file
        self.states = self.init_states()

        # Pad audio to be divisible by hop_size
        # This is to ensure we process all samples
        orig_len = audio.shape[-1]
        padding = (self.hop_size - (orig_len % self.hop_size)) % self.hop_size
        audio = torch.nn.functional.pad(audio, (0, padding))

        # Split audio into frames
        frames = audio.unfold(0, self.frame_size, self.hop_size)
        
        # Create a Hanning window for overlap-add
        window = torch.hann_window(self.frame_size)
        
        enhanced_frames = []
        
        print("Denoising audio...")
        for frame in tqdm(frames):
            windowed_frame = frame * window
            # Prepare input feed for ONNX session
            input_feed = {
                "input_frame": windowed_frame.numpy()[np.newaxis, :],
            }
            # Add states to the input feed
            for i, state in enumerate(self.states):
                # The state names might vary, but their order is consistent.
                # The input names from the session, excluding 'input_frame', correspond to the states.
                state_name = self.input_names[i+1]
                input_feed[state_name] = state

            # Run inference
            outputs = self.sess.run(self.output_names, input_feed)

            # The first output is the enhanced frame
            enhanced_frame = outputs[0]
            # Apply window again for synthesis
            enhanced_frames.append(torch.from_numpy(enhanced_frame).squeeze(0) * window)

            # The rest of the outputs are the updated states
            self.states = outputs[1:]

        # Overlap-add to reconstruct the audio
        enhanced_audio = torch.zeros(orig_len + padding)
        for i, frame in enumerate(enhanced_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            enhanced_audio[start:end] += frame

        # Trim to original length
        return enhanced_audio[:orig_len]


def main(args):
    """
    Main function to load model, process audio, and save the result.
    """
    try:
        # Load audio file
        noisy_audio, sr = torchaudio.load(args.input_path)
        print(f"Loaded audio from {args.input_path} with sample rate {sr}.")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Initialize the denoiser
    denoiser = OnnxDF(args.model_path)

    # Denoise the audio
    enhanced_audio = denoiser(noisy_audio.squeeze(0), sr)

    # Save the enhanced audio
    try:
        torchaudio.save(
            args.output_path,
            enhanced_audio.unsqueeze(0),
            sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        print(f"Saved enhanced audio to {args.output_path}")
    except Exception as e:
        print(f"Error saving audio file: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoise an audio file using a DeepFilterNet ONNX model.")
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        default="denoiser_model.ort", 
        help="Path to the denoiser ONNX/ORT model file."
    )
    parser.add_argument(
        "-i", "--input_path", 
        type=str, 
        required=True, 
        help="Path to the noisy input audio file (WAV format)."
    )
    parser.add_argument(
        "-o", "--output_path", 
        type=str, 
        default="enhanced_audio.wav", 
        help="Path to save the enhanced output audio file."
    )
    
    cli_args = parser.parse_args()
    main(cli_args)
