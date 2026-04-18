import json
import os
import argparse
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def open_image_rgb(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_prompt(question: str, is_reasoning: bool, options: Optional[List[str]] = None) -> str:
    if options:
        option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

        if is_reasoning:
            return (
                "Please first infer the geographic region or country based on the road infrastructure, "
                "traffic signs, and surrounding environment. Then, answer the question using only the image.\n"
                "Choose exactly one option.\n"
                f"Question: {question}\n"
                f"Options:\n{option_text}\n"
                "Output the explanation (include geographic country, related regional traffic rules, and visual reasoning) and the final answer.\n"
            )
        else:
            return (
                "Answer the question using only the image.\n"
                "Choose exactly one option.\n"
                f"Question: {question}\n"
                f"Options:\n{option_text}\n"
                "Output only the final answer.\n"
            )

    if is_reasoning:
        return (
            "Please first infer the geographic region or country based on the road infrastructure, "
            "traffic signs, and surrounding environment. Then, answer the question using only the image.\n"
            f"Question: {question}\n"
            "Output the explanation (include geographic country, related regional traffic rules, and visual reasoning) and the final answer.\n"
        )
    else:
        return (
            "Answer the question using only the image.\n"
            f"Question: {question}\n"
            "Output only the final answer.\n"
        )


def load_glm_model(
    model_path: str = "zai-org/GLM-4.6V-Flash",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | Dict[str, int] = "auto",
):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return processor, model


def load_qwen_model(
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | Dict[str, int] = "auto",
):
    processor = AutoProcessor.from_pretrained(model_path)
    if "Qwen3" in model_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    model.eval()
    return processor, model


def run_glm_inference(
    model: Glm4vForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """
    GLM-4.6V-Flash inference.
    Uses the chat-template style shown on the model page.
    """

    image_content = [
        {
            "type": "image",
            "image": image
        } for image in image_path
    ]


    messages = [
        {
            "role": "user",
            "content": image_content + [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return output_text


def run_qwen_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: list,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Qwen3-VL-8B-Instruct inference.
    Uses processor.apply_chat_template + processor(text=..., images=...).
    """
    image_content = [
        {
            "type": "image",
            "image": open_image_rgb(image)
        } for image in image_path
    ]

    messages = [
        {
            "role": "user",
            "content": image_content + [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[open_image_rgb(image) for image in image_path],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return output_text


def run_internvl_inference(pipe, images, text):
    if isinstance(images, list):
        images = [load_image(img_url) for img_url in images]
        prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(images))])
        print(prefix + text)
        response = pipe((prefix + text, images))
    else:
        image = load_image(images)
        response = pipe((text, image))

    return response.text


def infer_dataset(
    model_family: str,
    model,
    processor,
    data: Dict[str, Any],
    image_root: str = "",
    max_new_tokens: int = 64,
    is_reasoning: bool = False,
) -> List[Dict[str, Any]]:


    results = {}


    for country in tqdm(data.keys(), desc="Countries"):
        results[country] = []
        for i, line in tqdm(enumerate(data[country]), total=len(data[country])):
            image_path = line['images']
            question = line['question']
            answer = line['answer']
            options = line['options']

            prompt = build_prompt(question, is_reasoning=is_reasoning, options=options)

            if model_family == "glm":
                pred = run_glm_inference(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
            elif model_family == "internvl":
                pred = run_internvl_inference(model, image_path, prompt)
            elif model_family == "qwen":
                pred = run_qwen_inference(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )


            print("="*50)
            print(image_path[-1])
            print(question)
            print(pred)
            print(answer)

            results[country].append(
                {
                    "type": line['type'],
                    "task": line['task'],
                    "images": line['images'],
                    "country": country,
                    "question": line['question'],
                    "gt": answer,
                    "pred": pred,
                    "options": line['options'],
                    "rule": line['rule'],
                }
            )

    return results


def parse_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=["glm", "qwen", "internvl"],
        help="Which model family to run.",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--is_reasoning", type=bool, default=False)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    torch_dtype = parse_dtype(args.dtype)

    if args.model_family == "glm":
        model_path = args.model_path or "zai-org/GLM-4.6V-Flash"
        print(f"Loading GLM processor/model from {model_path} ...")
        processor, model = load_glm_model(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    elif args.model_family == "internvl":
        model_path = args.model_path or "OpenGVLab/InternVL3-8B"
        print(f"Loading InternVL processor/model from {model_path} ...")
        model = pipeline(args.model_path, backend_config=TurbomindEngineConfig(session_len=16384, tp=1, device='cuda'),
                        chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
        processor = None
    else:
        model_path = args.model_path or "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading Qwen processor/model from {model_path} ...")
        processor, model = load_qwen_model(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    print(f"Loading test data from {args.test_json} ...")
    data = load_json(args.test_json)

    print("Running inference ...")
    results = infer_dataset(
        model_family=args.model_family,
        model=model,
        processor=processor,
        data=data,
        image_root=args.image_root,
        max_new_tokens=args.max_new_tokens,
        is_reasoning=args.is_reasoning,
    )

    print(f"Saving results to {args.output_json} ...")
    save_json(results, args.output_json)
    print("Done.")


if __name__ == "__main__":
    main()