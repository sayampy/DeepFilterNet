import onnxruntime as ort
import json
import argparse


def generate_metadata(model_path: str):
    """
    Loads an ONNX model and extracts its metadata.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        dict: A dictionary containing the model's metadata.
    """
    try:
        sess = ort.InferenceSession(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None

    metadata = {}

    # --- Extract Input/Output Information ---
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    metadata["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs]
    metadata["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outputs]

    # --- Hardcoded values based on DeepFilterNet architecture ---
    # These are typically fixed for a given model architecture.
    metadata["hop_size"] = 512
    metadata["fft_size"] = 960
    metadata["frame_size"] = metadata["hop_size"]

    # --- Identify state inputs ---
    input_names = [i.name for i in inputs]
    state_input_names = [name for name in input_names if name not in ("input_frame", "atten_lim_db")]
    metadata["state_input_names"] = state_input_names
    has_single_state_tensor = "states" in state_input_names

    if has_single_state_tensor:
        states_shape = [s.shape for s in inputs if s.name == "states"][0]
        metadata["states_len"] = states_shape[0]

    return metadata


def main():
    """
    Main function to parse arguments and run metadata generation.
    """
    parser = argparse.ArgumentParser(description="Export ONNX model metadata to a JSON file.")
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the ONNX model file."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="model_metadata.json",
        help="Path to save the output JSON metadata file."
    )
    args = parser.parse_args()

    model_metadata = generate_metadata(args.model_path)

    if model_metadata:
        with open(args.output_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        print(f"Successfully exported metadata to {args.output_path}")


if __name__ == "__main__":
    main()
