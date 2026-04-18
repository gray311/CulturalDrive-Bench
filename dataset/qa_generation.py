import os
import re
import json
import boto3
from tqdm import tqdm
from PIL import Image, ImageDraw
from collections import defaultdict


from traffic_handbook import (
    china_traffic_handbook,
    japan_traffic_handbook,
    uk_traffic_handbook,
    india_traffic_handbook,
    us_traffic_handbook,
    singapore_traffic_handbook
)
from .template import state_extraction_prompt, qa_generation_prompt

def denormalize_box(box, img_w, img_h, box_format="xywh"):
    x, y, w, h = box

    if box_format == "xywh":
        x1 = x * img_w
        y1 = y * img_h
        x2 = (x + w) * img_w
        y2 = (y + h) * img_h
    elif box_format == "cxcywh":
        cx = x * img_w
        cy = y * img_h
        bw = w * img_w
        bh = h * img_h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
    else:
        raise ValueError(f"Unsupported box_format: {box_format}")

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return [x1, y1, x2, y2]


def validate_and_draw_boxes(
        image_path: str,
        sample: dict,
        save_dir: str,
        box_format: str = "xywh",
        max_objects: int = 50,
        line_width: int = 3
):
    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    object_info = sample.get("object_info", {})
    labels = object_info.get("labels", [])
    boxes = object_info.get("boxes", [])

    problems = []

    for i, (label, box) in enumerate(zip(labels[:max_objects], boxes[:max_objects])):
        if len(box) != 4:
            problems.append({
                "index": i,
                "label": label,
                "box": box,
                "reason": "box length is not 4"
            })
            continue

        if not all(isinstance(v, (int, float)) for v in box):
            problems.append({
                "index": i,
                "label": label,
                "box": box,
                "reason": "box contains non-numeric values"
            })
            continue

        if not all(0 <= v <= 1 for v in box):
            problems.append({
                "index": i,
                "label": label,
                "box": box,
                "reason": "box values are not normalized to [0,1]"
            })
            continue

        x1, y1, x2, y2 = denormalize_box(box, image.width, image.height, box_format=box_format)

        if x2 <= x1 or y2 <= y1:
            problems.append({
                "index": i,
                "label": label,
                "box": box,
                "reason": "invalid box after denormalization"
            })
            continue

        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
        draw.text((x1, max(0, y1 - 12)), f"{i}:{label}", fill="red")

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    image.save(save_path)

    return save_path, problems


def normalize_image_format(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mapping = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp",
    }
    if ext not in mapping:
        raise ValueError(f"Unsupported image extension: {ext}")
    return mapping[ext]


def build_multiframe_content(frame_paths, prompt: str):
    """
    Build Bedrock converse content with multiple images followed by text.
    """
    content = []

    for frame_path in frame_paths:
        with open(frame_path, "rb") as f:
            image_bytes = f.read()

        image_format = normalize_image_format(frame_path)

        content.append({
            "image": {
                "format": image_format,
                "source": {"bytes": image_bytes}
            }
        })

    content.append({"text": prompt})
    return content


def call_qwen_with_image(image_path, prompt: str, max_tokens: int = 4096) -> str:
    frame_paths = image_path
    content = build_multiframe_content(frame_paths, prompt)

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.2,
            "topP": 0.9
        }
    )

    content = response["output"]["message"]["content"]
    texts = [block["text"] for block in content if "text" in block]
    return "\n".join(texts).strip()


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_handbook(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def xywh_norm_to_xyxy_1000(box):
    x, y, w, h = box
    x1 = round(x * 1000)
    y1 = round(y * 1000)
    x2 = round((x + w) * 1000)
    y2 = round((y + h) * 1000)

    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    return [x1, y1, x2, y2]

def build_objects_json(sample, w, h, max_objects=50):
    object_info = sample.get("object_info", {})
    labels = object_info.get("labels", [])
    boxes = object_info.get("boxes", [])

    objects = []
    for i, (label, bbox) in enumerate(zip(labels[:max_objects], boxes[:max_objects])):
        if len(bbox) != 4:
            continue
        objects.append({
            "id": i,
            "label": label,
            "bbox_2d": xywh_norm_to_xyxy_1000(bbox)
        })

    return json.dumps(objects, ensure_ascii=False, separators=(",", ":"))


def filter_complex_samples(data, min_labels: int = 5):
    results = []
    if "1" in data.keys():
        data = data["1"]
    for sample in data:
        labels = sample.get("object_info", {}).get("labels", [])
        if len(labels) >= min_labels:
            results.append(sample)
    return results


def check_existing_path(output_json, country):
    existing_paths = set()
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
            results = results[country]
            existing_paths = {x.get("image_path") for x in results if "image_path" in x}
            print(f"Resume from existing file: {len(results)} samples already saved")
        except Exception:
            print("Warning: failed to load existing output file, starting fresh")
            results = []
            existing_paths = set()

    return existing_paths


def annotate_samples(data, output_json: str, checkpoint_every: int = 50):
    results = defaultdict(list)

    for country in tqdm(data.keys(), desc="Annotating Countries"):
        samples = data[country]
        existing_paths = check_existing_path(output_json, country)
        if country == "cn":
            handbook_text = china_traffic_handbook
        elif country == "us":
            handbook_text = us_traffic_handbook
        elif country == "uk":
            handbook_text = uk_traffic_handbook
        elif country == "jp":
            handbook_text = japan_traffic_handbook
        elif country == "ind":
            handbook_text = india_traffic_handbook
        elif country == "sg":
            handbook_text = singapore_traffic_handbook

        for idx, sample in enumerate(tqdm(samples)):
            image_path = sample.get("image_path", "")

            if image_path[-1] in existing_paths:
                continue

            if not image_path[-1] or not os.path.exists(image_path[-1]):
                print(f"Skip missing image: {image_path[-1]}")
                continue

            image = Image.open(image_path[-1])
            image.save("1.png")
            w, h = image.size
            labels = sample.get("object_info", {}).get("labels", [])

            #
            # debug_img_path, problems = validate_and_draw_boxes(
            #     image_path=image_path,
            #     sample=sample,
            #     save_dir="/weka/home/ext-yingzima/CulturalDrive/debug_boxes",
            #     box_format="xywh",
            #     max_objects=50
            # )

            # objects_str = build_objects_json(sample, w, h, max_objects=50)
            handbook = sample.get("handbook", handbook_text)
            prompt = qa_generation_prompt.replace("<state>", sample['annotation']).replace("<country>", country).replace("<handbook>", handbook)

            try:
                explanation = call_qwen_with_image(image_path, prompt)
                new_sample = dict(sample)
                new_sample["annotation"] = explanation
                new_sample["scene_complexity"] = {
                    "is_complex_scene": True,
                    "num_labels": len(labels)
                }

                results[country].append(new_sample)

            except Exception as e:
                print(f"Error on {image_path}: {e}")
                continue

            if len(results[country]) % checkpoint_every == 0:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: {len(results)} samples")



    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


INPUT_JSON = "/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_scenarios_state_V1.json"
OUTPUT_JSON = "/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_scenarios_qa.json"
AWS_REGION = os.environ["AWS_REGION"]
MODEL_ID = "qwen.qwen3-vl-235b-a22b"
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def main():
    data = load_data(INPUT_JSON)

    print(data.keys())

    results = annotate_samples(
        data,
        output_json=OUTPUT_JSON,
        checkpoint_every=20
    )

    print("Done.")
    print("Saved to:", OUTPUT_JSON)
    print("Annotated samples:", len(results))


if __name__ == "__main__":
    main()