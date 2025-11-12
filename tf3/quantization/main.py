from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path_to_your_model")
    parser.add_argument("--output_path", type=str, default="path_to_your_output")
    return parser.parse_args()

def quantize_model(model_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,             # or load_in_8bit=True
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return model


if __name__ == "__main__":
    args = parse_args()
    model = quantize_model(args.model_path)
    model.save_pretrained(args.output_path)
