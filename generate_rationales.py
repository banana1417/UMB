import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from rationale_prompt import RATIONALE_PROMPT, parse_rationales


def generate(model, processor, title, image_path, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": f"News title: {title}\n\n{RATIONALE_PROMPT}"},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_rationales(text)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for row in rows:
        row["true_reason"], row["false_reason"] = generate(
            model, processor, row["title"], row["image_path"], device
        )
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON list without rationale fields")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    main(parser.parse_args())
