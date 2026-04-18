import os
import re
import json
import boto3
from PIL import Image
from tqdm import tqdm
from traffic_handbook import japan_traffic_handbook

from PIL import Image, ImageDraw

from .template import PROMPT_TEMPLATE

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



def get_temporal_frame_paths(
    image_path: str,
    num_frames: int = 3,
    step: int = 2,
    interval_ms: int = 500,
):
    """
    Unified temporal sampler supporting:
    1. index-based: 0028.jpg
    2. index+suffix: 078_CAM_JOINT.jpg
    3. timestamp-based: 1616343528200.jpg

    Behavior:
    - If all requested temporal frames exist, return them old -> new
    - If any historical frame is missing, return [image_path] only
    """

    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    stem, ext = os.path.splitext(filename)

    match = re.match(r"^(\d+)(.*)$", stem)
    if not match:
        return [image_path]

    numeric_part = match.group(1)
    suffix = match.group(2)
    value = int(numeric_part)

    is_timestamp = len(numeric_part) >= 12 or value > 1e10

    frame_paths = []

    if is_timestamp:
        for i in reversed(range(num_frames)):
            target = value - interval_ms * i
            frame_name = f"{target}{suffix}{ext}"
            frame_path = os.path.join(directory, frame_name)

            if not os.path.exists(frame_path):
                return [image_path]

            frame_paths.append(frame_path)
    else:
        width = len(numeric_part)

        for i in reversed(range(num_frames)):
            idx = value - step * i
            if idx < 0:
                return [image_path]

            frame_name = f"{idx:0{width}d}{suffix}{ext}"
            frame_path = os.path.join(directory, frame_name)

            if not os.path.exists(frame_path):
                return [image_path]

            frame_paths.append(frame_path)

    return frame_paths

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


def call_qwen_with_image(image_path: str, prompt: str, max_tokens: int = 2048) -> str:
    frame_paths = get_temporal_frame_paths(image_path, num_frames=3, step=2)

    print("Using frames:", frame_paths)
    print(prompt)

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


def annotate_samples(samples, output_json: str, handbook_text: str, checkpoint_every: int = 50):
    results = []

    existing_paths = set()
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
            existing_paths = {x.get("image_path") for x in results if "image_path" in x}
            print(f"Resume from existing file: {len(results)} samples already saved")
        except Exception:
            print("Warning: failed to load existing output file, starting fresh")
            results = []
            existing_paths = set()

    for idx, sample in enumerate(tqdm(samples)):
        image_path = sample.get("image_path", "")

        if image_path in existing_paths:
            continue

        if not image_path or not os.path.exists(image_path):
            print(f"Skip missing image: {image_path}")
            continue

        image = Image.open(image_path)
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

        objects_str = build_objects_json(sample, w, h, max_objects=50)

        country = sample.get("country", DEFAULT_COUNTRY)
        handbook = sample.get("handbook", handbook_text)

        prompt = PROMPT_TEMPLATE.replace("<objects>", objects_str).replace("<country>", country).replace("<handbook>", handbook)

        try:
            explanation = call_qwen_with_image(image_path, prompt)

            new_sample = dict(sample)
            new_sample["annotation"] = explanation
            new_sample["scene_complexity"] = {
                "is_complex_scene": True,
                "num_labels": len(labels)
            }

            results.append(new_sample)

        except Exception as e:
            print(f"Error on {image_path}: {e}")
            continue

        if len(results) % checkpoint_every == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved: {len(results)} samples")

        print(explanation)


    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


INPUT_JSON = "/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/cn_scenario.json"
OUTPUT_JSON = "/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/CultureDrive/cn_scenario_annotations_mini.json"
AWS_REGION = os.environ["AWS_REGION"]
MODEL_ID = "qwen.qwen3-vl-235b-a22b"
DEFAULT_COUNTRY = "uk"
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def main():
    data = load_data(INPUT_JSON)
    handbook_text = japan_traffic_handbook

    print("Total samples:", len(data))

    complex_samples = filter_complex_samples(data, min_labels=15)
    print("Complex scenes (labels >= 20):", len(complex_samples))

    results = annotate_samples(
        complex_samples,
        output_json=OUTPUT_JSON,
        handbook_text=handbook_text,
        checkpoint_every=50
    )

    print("Done.")
    print("Saved to:", OUTPUT_JSON)
    print("Annotated samples:", len(results))


if __name__ == "__main__":
    main()