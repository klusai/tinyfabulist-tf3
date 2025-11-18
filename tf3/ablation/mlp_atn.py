import argparse
import json
import math
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tf3.ablation.finetune import finetune_model


DEFAULT_TEST_TEXTS = [
    # Ultra-short fable prompts
    "Vulpea si",
    "Lupul si",
    "Ursul si",
    "Gaina cu",
    "Broasca si",
    "Furnica si",
    "Soarecele si",
    "Corbul cu",
    "Iepurele si",
    "Leul si",
    
    # Classic fable starters
    "A fost odata",
    "Intr-o padure",
    "Pe un camp",
    "La margine",
    "In gradina",
    "Sub un copac",
    "Langa un rau",
    "Intr-un sat",
    "La țară",
    "In impărăție",
    
    # Animal actions
    "Vulpea fugărea",
    "Lupul vâna",
    "Ursul dormea",
    "Gaina cânta",
    "Broasca înota",
    "Furnica munci",
    "Soarecele fura",
    "Corbul zboara",
    "Iepurele alerga",
    "Leul poruncea",
    
    # Fable objects/items
    "O bucata de",
    "O ramura de",
    "Un ou de",
    "O pana de",
    "O piatra prețioasă",
    "O coroniță de",
    "O comoara ascunsă",
    "O vrajă veche",
    "O invitație la",
    "O promisiune de",
    
    # Moral concepts
    "Lenea duce",
    "Munca aduce",
    "Cine alearga",
    "Vorba dulce",
    "Gândirea lentă",
    "Increderea oarbă",
    "Lăcomia distruge",
    "Smerenia castigă",
    "Prietenia adevarată",
    "Minciuna are",
    
    # Complete mini-fables (1 sentence)
    "Vulpea a pacalit corbul.",
    "Furnica a strans pentru iarna.",
    "Broasca a invins iepurele.",
    "Lupul a mancat oaia.",
    "Ursul a gasit miere.",
    "Gaina a dat ouă de aur.",
    "Soarecele a salvat leul.",
    "Corbul a pierdut branza.",
    "Iepurele a pierdut cursa.",
    "Leul a impartit prada.",
    
    # Dialog starters
    "'Buna ziua,' spuse vulpea.",
    "'Ajuta-ma!' striga furnica.",
    "'Vino repede!' chema iepurele.",
    "'Asculta-ma!' porunci leul.",
    "'Am gasit!' exclamă ursul.",
    "'Am pierdut!' plânse corbul.",
    "'Am castigat!' anunta broasca.",
    "'Am invatat!' recunoscu lupul.",
    "'Am promis!' jura soarecele.",
    "'Am muncit!' spuse furnica.",
    
    # Weather/season context
    "Intr-o zi frumoasa",
    "Cand soarele stralucea",
    "In timpul iernii",
    "Primavara venise",
    "Vantul batea",
    "Ploaia canta",
    "Zapada acoperea",
    "Frunzele cadeau",
    "Flori infloreau",
    "Fructele maturau",
    
    # Simple conflicts
    "Cine e mai puternic?",
    "Cine alearga mai repede?",
    "Cine e mai destept?",
    "Cine castiga comoara?",
    "Cine ajunge primul?",
    "Cine gaseste raspunsul?",
    "Cine invata lectia?",
    "Cine spune adevarul?",
    "Cine respecta promisiunea?",
    "Cine ajuta prietenul?",
    
    # Ultra-short morals
    "Lenea nu plateste.",
    "Munca este rasplatita.",
    "Grabeste-te incet.",
    "Cine astepa ajunge.",
    "Minte si vei prinde.",
    "Increderea ucide.",
    "Puterea nu inseamna tot.",
    "Desteptul castiga.",
    "Rabdarea are roade.",
    "Prietenia adevarata dureaza.",
]

REPORT_PROMPTS = [
    "Vulpea si",
    "Intr-o padure",
    "Broasca a invins iepurele.",
    "Lupul a mancat oaia.",
    "Lenea nu plateste.",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_rate(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def normalize_report_prompts(custom_prompts):
    if not custom_prompts:
        return REPORT_PROMPTS[:5]
    prompts = list(custom_prompts)
    if len(prompts) < 5:
        prompts = (prompts + REPORT_PROMPTS)[:5]
    else:
        prompts = prompts[:5]
    return prompts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="klusai/tf3-50m-base")
    parser.add_argument("--mlp_ablation_rate", type=float, default=0.1)
    parser.add_argument("--attention_head_ablation_rate", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default="artifacts/tf3-50m-base-ab-mlp-atn")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--eval_text",
        action="append",
        help="Custom evaluation text. Repeat the flag to add multiple entries.",
    )
    parser.add_argument(
        "--benchmark_table_path",
        type=str,
        default="artifacts/ablation.md",
        help="Destination file for the markdown benchmark table.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="artifacts/ablation_report.json",
        help="Path where the JSON benchmark report will be stored.",
    )
    parser.add_argument(
        "--report_prompts",
        type=str,
        nargs="+",
        help="Custom prompts (max 5) for sampling ablated models in the JSON report.",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Persist ablated checkpoints for each configuration.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="Fine-tune ablated models after ablation operations.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=3,
        help="Number of epochs for fine-tuning (if --finetune is enabled).",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=8,
        help="Batch size for fine-tuning.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=5e-4,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--finetune_max_samples",
        type=int,
        default=20000,
        help="Maximum samples from dataset for fine-tuning.",
    )
    return parser.parse_args()

def get_device(model):
    return next(model.parameters()).device

def build_mlp_mask(mlp, p_disable: float, device):
    H = mlp.down_proj.in_features   # intermediate_size, e.g. 1365
    k = int(p_disable * H)
    mask = torch.ones(H, device=device)
    if k > 0:
        idx = torch.randperm(H, device=device)[:k]
        mask[idx] = 0.0
    return mask.view(1, 1, H)  # [1,1,H] for broadcasting


def mlp_ablation_pre_hook(mask):
    # mask: [1,1,H]
    def hook(module, inputs):
        # module is down_proj; inputs[0] is h: [B, S, H]
        h = inputs[0]
        h_masked = h * mask
        # forward_pre_hook expects a tuple of inputs
        return (h_masked, *inputs[1:])
    return hook


def apply_mlp_ablation(model, p_disable: float):
    """
    Randomly ablate p_disable fraction of MLP neurons in every layer.
    Returns list of hook handles (call .remove() to disable).
    """
    device = get_device(model)
    handles = []
    for layer in model.model.layers:
        mlp = layer.mlp
        mask = build_mlp_mask(mlp, p_disable, device)
        h = mlp.down_proj.register_forward_pre_hook(mlp_ablation_pre_hook(mask))
        handles.append(h)
    return handles

def build_head_mask(num_heads: int, head_dim: int, p_disable: float, device):
    """
    Build a [1,1,hidden_size] mask that zeroes selected heads.
    """
    H = num_heads * head_dim
    assert H > 0
    num_disable = int(p_disable * num_heads)

    head_mask = torch.ones(num_heads, device=device)
    if num_disable > 0:
        idx = torch.randperm(num_heads, device=device)[:num_disable]
        head_mask[idx] = 0.0          # 0 = ablated head, 1 = active

    # expand each head's scalar mask over its head_dim
    mask = head_mask.repeat_interleave(head_dim)  # [H]
    return mask.view(1, 1, H)                     # [1,1,hidden_size]


def attn_proj_ablation_hook(mask):
    # mask over last dim of q/k/v projection outputs
    def hook(module, inputs, output):
        # output: [B, S, hidden_size]
        return output * mask
    return hook


def apply_attention_head_ablation(model, p_disable: float):
    """
    Randomly ablate p_disable fraction of attention heads in every layer
    by masking Q, K, and V projections.
    Returns list of hook handles.
    """
    device = get_device(model)
    handles = []

    # assume all layers share same num_heads + head_dim
    sample_attn = model.model.layers[0].self_attn
    num_heads = getattr(sample_attn, "num_heads", model.config.num_attention_heads)
    head_dim = getattr(sample_attn, "head_dim", model.config.hidden_size // num_heads)
    hidden_size = num_heads * head_dim
    assert hidden_size == model.config.hidden_size

    mask = build_head_mask(num_heads, head_dim, p_disable, device)

    for layer in model.model.layers:
        attn = layer.self_attn
        # hook on q, k, v
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            h = proj.register_forward_hook(attn_proj_ablation_hook(mask))
            handles.append(h)

    return handles

def apply_ablation(model, mlp_p_disable=0.0, head_p_disable=0.0):
    hooks = []
    if mlp_p_disable > 0.0:
        hooks += apply_mlp_ablation(model, mlp_p_disable)
    if head_p_disable > 0.0:
        hooks += apply_attention_head_ablation(model, head_p_disable)
    return hooks

def test_generation(model, tokenizer, prompt, max_new_tokens):
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    return generated_text


def evaluate_cross_entropy(model, tokenizer, texts):
    device = get_device(model)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for idx, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            inputs.pop("token_type_ids", None)
            input_ids = inputs["input_ids"]

            if input_ids.size(1) < 2:
                print(f"[skip] Text too short for evaluation: '{text}'")
                continue

            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss
            tokens = input_ids.size(1) - 1
            total_loss += loss.item() * tokens
            total_tokens += tokens

    if total_tokens == 0:
        return None, None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_samples(model, tokenizer, prompts, max_new_tokens):
    samples = []
    for prompt in prompts:
        text = test_generation(model, tokenizer, prompt, max_new_tokens)
        samples.append({"prompt": prompt, "output": text})
    return samples


def write_json_report(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_markdown_table(results, base_loss, base_ppl):
    if not results or base_loss is None or base_ppl is None:
        return None

    lines = [
        "# Ablation Benchmark Summary",
        "",
        f"- Baseline loss: {base_loss:.4f}",
        f"- Baseline perplexity: {base_ppl:.2f}",
        "",
        "| MLP Rate | Attn Rate | Abl Loss | Abl PPL | ΔLoss% | ΔPPL% |",
        "|--------:|----------:|---------:|--------:|-------:|------:|",
    ]

    for entry in results:
        delta_loss = entry["delta_loss"]
        delta_ppl = entry["delta_ppl"]
        lines.append(
            (
                f"| {entry['mlp_rate']:.2f} "
                f"| {entry['attn_rate']:.2f} "
                f"| {entry['abl_loss']:.4f} "
                f"| {entry['abl_ppl']:.2f} "
                f"| {format_delta_cell(delta_loss)} "
                f"| {format_delta_cell(delta_ppl)} |"
            )
        )

    return "\n".join(lines)


def save_markdown_table(table_md, path):
    if not table_md:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table_md + "\n")


MAX_DELTA_INTENSITY = 40.0
COLOR_IMPROVE = (52, 168, 83)  # Google green
COLOR_REGRESS = (234, 67, 53)  # Google red


def lerp_color(target_rgb, intensity):
    return tuple(int(255 + (c - 255) * intensity) for c in target_rgb)


def color_for_delta(delta):
    if delta is None:
        return "#ffffff"
    base = COLOR_IMPROVE if delta < 0 else COLOR_REGRESS
    norm = min(abs(delta) / MAX_DELTA_INTENSITY, 1.0)
    r, g, b = lerp_color(base, norm)
    return f"#{r:02x}{g:02x}{b:02x}"


def format_delta_cell(delta):
    if delta is None:
        return "N/A"
    color = color_for_delta(delta)
    return f'<span style="background-color:{color};padding:0 6px;border-radius:4px;">{delta:+.2f}%</span>'


def run_pipeline(args):            
    print("Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    base_model.resize_token_embeddings(len(tokenizer))

    eval_texts = args.eval_text if args.eval_text else DEFAULT_TEST_TEXTS
    report_prompts = normalize_report_prompts(args.report_prompts)

    print("\n[Original model results]")
    base_loss, base_ppl = evaluate_cross_entropy(base_model, tokenizer, eval_texts)
    print(f"Base loss: {base_loss:.4f}, Base ppl: {base_ppl:.2f}")

    mlp_abblation_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    attention_head_ablation_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    benchmark_rows = []
    for mlp_ablation_rate in mlp_abblation_rates:
        for attention_head_ablation_rate in attention_head_ablation_rates:
            print(f"Running pipeline for MLP ablation rate {mlp_ablation_rate} and attention head ablation rate {attention_head_ablation_rate}")
            ablated_model = AutoModelForCausalLM.from_pretrained(args.model_path)
            ablated_model.resize_token_embeddings(len(tokenizer))
            apply_ablation(ablated_model, mlp_ablation_rate, attention_head_ablation_rate)

            save_dir = None
            finetuned_dir = None
            
            # Fine-tune if requested
            if args.finetune:
                print(f"\n[Fine-tuning ablated model: MLP={mlp_ablation_rate:.2f}, Attn={attention_head_ablation_rate:.2f}]")
                suffix = f"mlp_{sanitize_rate(mlp_ablation_rate)}__attn_{sanitize_rate(attention_head_ablation_rate)}"
                finetuned_dir = os.path.join(args.output_path, suffix + "-finetuned")
                
                # Fine-tune the ablated model
                finetune_model(
                    model=ablated_model,
                    tokenizer=tokenizer,
                    output_path=finetuned_dir,
                    batch_size=args.finetune_batch_size,
                    grad_accum_steps=4,
                    max_epochs=args.finetune_epochs,
                    base_lr=args.finetune_lr,
                    max_length=128,
                    max_samples=args.finetune_max_samples,
                )
                
                # Reload fine-tuned model for evaluation
                ablated_model = AutoModelForCausalLM.from_pretrained(finetuned_dir)
                ablated_model.resize_token_embeddings(len(tokenizer))
                # Re-apply ablation hooks (they were lost after fine-tuning)
                apply_ablation(ablated_model, mlp_ablation_rate, attention_head_ablation_rate)
            
            # Evaluate ablated (and optionally fine-tuned) model
            ablated_loss, ablated_ppl = evaluate_cross_entropy(ablated_model, tokenizer, eval_texts)
            samples = generate_samples(ablated_model, tokenizer, report_prompts, args.max_new_tokens)
            
            if args.save_models:
                suffix = f"mlp_{sanitize_rate(mlp_ablation_rate)}__attn_{sanitize_rate(attention_head_ablation_rate)}"
                save_dir = os.path.join(args.output_path, suffix)
                ensure_dir(save_dir)
                ablated_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

            if base_loss is not None and ablated_loss is not None:
                delta_loss = ((ablated_loss - base_loss) / base_loss) * 100
                delta_ppl = ((ablated_ppl - base_ppl) / base_ppl) * 100
                print(f"Original loss: {base_loss:.4f} | ppl: {base_ppl:.2f}")
                print(f"Ablated loss:  {ablated_loss:.4f} | ppl: {ablated_ppl:.2f}")
                print(f"Δ loss: {delta_loss:+.2f}% | Δ ppl: {delta_ppl:+.2f}%")
                print("--------------------------------")
                print()
            else:
                delta_loss = None
                delta_ppl = None

            benchmark_rows.append(
                {
                    "mlp_rate": mlp_ablation_rate,
                    "attn_rate": attention_head_ablation_rate,
                    "abl_loss": ablated_loss,
                    "abl_ppl": ablated_ppl,
                    "delta_loss": delta_loss,
                    "delta_ppl": delta_ppl,
                    "samples": samples,
                    "saved_model_dir": save_dir,
                    "finetuned_model_dir": finetuned_dir,
                }
            )

    table_md = build_markdown_table(benchmark_rows, base_loss, base_ppl)
    if table_md:
        print("\nMarkdown benchmark summary:\n")
        print(table_md)
        save_markdown_table(table_md, args.benchmark_table_path)
        print(f"\nSaved table to {args.benchmark_table_path}")

    report_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_path": args.model_path,
        "evaluation_texts": eval_texts,
        "report_prompts": report_prompts,
        "base_metrics": {"loss": base_loss, "ppl": base_ppl},
        "results": [
            {
                "mlp_rate": row["mlp_rate"],
                "attn_rate": row["attn_rate"],
                "ablated_loss": row["abl_loss"],
                "ablated_ppl": row["abl_ppl"],
                "delta_loss_pct": row["delta_loss"],
                "delta_ppl_pct": row["delta_ppl"],
                "samples": row["samples"],
                "saved_model_dir": row["saved_model_dir"],
            }
            for row in benchmark_rows
        ],
    }
    write_json_report(args.report_path, report_payload)
    print(f"Saved JSON report to {args.report_path}")


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()