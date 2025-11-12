from mlx_lm import load, quantize, save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path_to_your_model")
    parser.add_argument("--output_path", type=str, default="path_to_your_output")
    return parser.parse_args()

def quantize(model_path, output_path)
    # Load a model (any supported HF checkpoint)
    model, tokenizer = load(f"{model_path}")

    # Quantize to 6 bits
    quantized_model = quantize(model, bits=6)

    # Save the quantized version
    save(quantized_model, tokenizer, path=f"{output_path}")

if __name__ == "__main__":
    args = parse_args()
    model = quantize_model(args.model_path, args.output_path)
