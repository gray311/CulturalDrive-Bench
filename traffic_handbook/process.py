import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional
import re
import random


countries = ['sg', 'cn', 'ind', 'jp', 'uk', 'us']
random.seed(233)
outputs = defaultdict(list)


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


def extract_dataset_info(image_path: str) -> Dict[str, Optional[str]]:

    norm_path = os.path.normpath(image_path)
    parts = norm_path.split(os.sep)
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    dataset = None
    scene_id = None

    # 1) CoVLA-Dataset
    # .../CoVLA-Dataset/images/<scene_id>/<frame>.jpg
    if "CoVLA-Dataset" in parts:
        dataset = "CoVLA-Dataset"
        try:
            idx = parts.index("images")
            scene_id = parts[idx + 1]
        except Exception:
            scene_id = None

    # 2) idd_multimodal
    # .../idd_multimodal/primary/<scene_id>/leftCamImgs/<frame>.jpg
    elif "idd_multimodal" in parts:
        dataset = "idd_multimodal"
        try:
            idx = parts.index("primary")
            scene_id = parts[idx + 1]
        except Exception:
            scene_id = None

    # 3) once
    # .../once/data/<scene_id>/cam01/<frame>.jpg
    elif "once" in parts:
        dataset = "once"
        try:
            idx = parts.index("data")
            scene_id = f"{parts[idx + 1]}/{parts[idx + 2]}"
        except Exception:
            scene_id = None

    # 4) waymo
    # .../waymo/train/<scene_id>/<frame>.jpg
    elif "waymo" in parts:
        dataset = "waymo"
        try:
            if "train" in parts:
                idx = parts.index("train")
                scene_id = parts[idx + 1]
            elif "val" in parts:
                idx = parts.index("val")
                scene_id = parts[idx + 1]
            elif "test" in parts:
                idx = parts.index("test")
                scene_id = parts[idx + 1]
        except Exception:
            scene_id = None

    # 5) lingoqa
    # .../lingoqa/train/<scene_id>/<frame>.jpg
    elif "lingoqa" in parts:
        dataset = "lingoqa"
        try:
            if "train" in parts:
                idx = parts.index("train")
                scene_id = parts[idx + 1]
            elif "val" in parts:
                idx = parts.index("val")
                scene_id = parts[idx + 1]
            elif "test" in parts:
                idx = parts.index("test")
                scene_id = parts[idx + 1]
        except Exception:
            scene_id = None

    # 6) NuScenes
    # .../nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883719412465.jpg
    elif "nuscenes" in parts:
        dataset = "nuscenes"
        try:
            scene_id = parts[-1].split("__")[0]
            stem = parts[-1].split("__")[-1]
        except Exception:
            scene_id = None

    else:
        dataset = "unknown"
        if len(parts) >= 2:
            scene_id = parts[-2]

    frame_id_raw = stem

    # 0013 -> 13
    # 1619665112000 -> 1619665112000
    # 032_CAM_JOINT -> 32
    m = re.match(r"^(\d+)", stem)
    frame_id_numeric = int(m.group(1)) if m else None

    return {
        "dataset": dataset,
        "scene_id": scene_id,
        "frame_id_raw": frame_id_raw,
        "frame_id_numeric": frame_id_numeric,
    }


def get_frame_path(country, image_path):
    frame_path = None
    # 2Hz
    if country == "jp":
        # 2Hz
        frame_path =  get_temporal_frame_paths(image_path, num_frames=3, step=1)

    elif country == "cn":
        # 2Hz
        frame_path =  get_temporal_frame_paths(image_path, num_frames=3, interval_ms=500)

    elif country == "uk":
        # 1Hz
        frame_path =  get_temporal_frame_paths(image_path, num_frames=3, step=1)

    elif country == "us":
        # 10hz
        frame_path =  get_temporal_frame_paths(image_path, num_frames=3, step=5)

    elif country == "ind":
        # 15Hz
        frame_path =  get_temporal_frame_paths(image_path, num_frames=3, step=7)

    from PIL import Image
    for path in frame_path:
        img = Image.open(path)

    return frame_path


def filter_complex_samples(data, min_labels: int = 10):
    results = []
    for sample in data:
        labels = sample.get("object_info", {}).get("labels", [])
        if len(labels) >= min_labels:
            results.append(sample)

    return results

cnt = 0
new_data = defaultdict(list)
for country in countries:
    with open(f"{country}_scenario.json", 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "1" in data.keys():
            data = data["1"]

    if country == "us":
        data = filter_complex_samples(data, min_labels=20)
    elif country == "jp":
        data = filter_complex_samples(data, min_labels=16)
    elif country == "uk":
        data = filter_complex_samples(data, min_labels=14)

    data = filter_complex_samples(data, min_labels=10)

    outputs = []
    seen = defaultdict(list)
    for line in data:
        if not isinstance(line['image_path'], list):
            frame_path = get_frame_path(country, line['image_path'])
        else:
            frame_path = line['image_path']

        info = extract_dataset_info(frame_path[-1])

        # 2Hz
        gap = 0
        if country == "jp":
            # 2Hz
            gap = 12

        elif country == "cn":
            # 2Hz, interval_ms 500
            gap = 2000

        elif country == "uk":
            # 1Hz
            gap = 3

        elif country == "us":
            # 10Hz
            gap = 60

        elif country == "ind":
            # 15Hz
            gap = 20

        elif country == "sg":
            gap = 500000

        frames = seen[info['scene_id']]

        conflict = False
        for frame_seen in frames:
            if abs(info['frame_id_numeric'] - frame_seen) <= gap:
                conflict = True
                break

        if conflict:
            continue

        seen[info['scene_id']].append(info['frame_id_numeric'])

        line['image_path'] = frame_path
        outputs.append(line)


    # print(f"\n\n{country}: {len(data)}")
    print(f"\n\n{country}: {len(outputs)}")
    cnt += len(outputs)
    new_data[country] = outputs

    # print(data[0])

print(cnt)
with open(f"/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_scenarios.json", 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)