import os
import sys
import shutil
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
from collections import defaultdict
from typing import List
import argparse
import cv2
import copy
import json
import pickle
import descartes
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def find_nearest_box(boxes, agent_box):
    min_distance = float('inf')
    nearest_box = None

    boxes = [box for box in boxes if area(box) > 10000]

    for box in boxes:
        distance = calculate_distance(box, agent_box)
        if distance < min_distance:
            min_distance = distance
            nearest_box = box

    return nearest_box


def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    if inter == 0:
        return 0.0
    union = area(boxA) + area(boxB) - inter
    return inter / min(area(boxA), area(boxB))

def filter_overlap_keep_smaller(boxes: List[List[float]], labels: List[str], iou_thresh=0.5):
    assert len(boxes) == len(labels)
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            if iou(boxes[i], boxes[j]) > iou_thresh:
                if area(boxes[i]) <= area(boxes[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    filtered_boxes = [b for b, k in zip(boxes, keep) if k]
    filtered_labels = [l for l, k in zip(labels, keep) if k]
    return filtered_boxes, filtered_labels


def dino_detect_object(image, text, grounding_model, processor):
    inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_model.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    return results[0]["boxes"].cpu().numpy().tolist(), results[0]["labels"], results[0]['scores']


def deim_detect_object(image, text, grounding_model, processor):
    IMAGE_SIZE = (640, 640)
    CONFIDENCE_THRESHOLD = 0.4

    label_map = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
        15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
        20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
        25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
        30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
        35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
        40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
        45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
        50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
        55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
        60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
        65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
        70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
        75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
        80: 'toothbrush'
    }

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0).cuda()

    grounding_model.cuda()
    grounding_model.eval()
    with torch.no_grad():
        outputs = grounding_model(input_tensor, orig_target_sizes=torch.tensor([IMAGE_SIZE]).cuda())

    class_labels, bboxes, scores = outputs[0]["labels"], outputs[0]["boxes"], outputs[0]["scores"]

    detections = []
    for label, bbox, score in zip(class_labels, bboxes, scores):
        if score.item() >= CONFIDENCE_THRESHOLD:
            detection = {
                "label": label_map[label.item() + 1],
                "bounding_box": [
                    bbox[0].item() / IMAGE_SIZE[0],
                    bbox[1].item() / IMAGE_SIZE[1],
                    (bbox[2] - bbox[0]).item() / IMAGE_SIZE[0],
                    (bbox[3] - bbox[1]).item() / IMAGE_SIZE[1]
                ],
                "confidence": score.item()
            }
            detections.append(detection)

    return [item['bounding_box'] for item in detections], [item['label'] for item in detections], [item['confidence'] for item in detections]



def build_model(grounding_model_name="dino"):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    processor = None

    # init grounding dino model from huggingface
    if grounding_model_name == "dino":
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    elif grounding_model_name == "deim":
        import torch.nn as nn
        from huggingface_hub import PyTorchModelHubMixin

        from engine.backbone import HGNetv2, DINOv3STAs
        from engine.deim import HybridEncoder, LiteEncoder
        from engine.deim import DFINETransformer, DEIMTransformer
        from engine.deim.postprocessor import PostProcessor

        # There is an example in the end!

        class DEIMv2(nn.Module, PyTorchModelHubMixin):
            def __init__(self, config):
                super().__init__()
                if 'DINOv3STAs' in config:
                    self.backbone = DINOv3STAs(**config["DINOv3STAs"])
                else:
                    self.backbone = HGNetv2(**config["HGNetv2"])
                if 'LiteEncoder' in config:
                    self.encoder = LiteEncoder(**config["LiteEncoder"])
                else:
                    self.encoder = HybridEncoder(**config["HybridEncoder"])
                if 'DEIMTransformer' in config:
                    self.decoder = DEIMTransformer(**config["DEIMTransformer"])
                else:
                    self.decoder = DFINETransformer(**config["DFINETransformer"])
                self.postprocessor = PostProcessor(**config["PostProcessor"])

            def forward(self, x, orig_target_sizes):
                x = self.backbone(x)
                x = self.encoder(x)
                x = self.decoder(x)
                x = self.postprocessor(x, orig_target_sizes)

                return x


        deimv2_x_config = {
            "DINOv3STAs": {
                "name": "dinov3_vits16plus",
                "embed_dim": 256,
                "interaction_indexes": [
                    5,
                    8,
                    11
                ],
                "num_heads": None,
                "conv_inplane": 64,
                "hidden_dim": 256
            },
            "HybridEncoder": {
                "in_channels": [
                    256,
                    256,
                    256
                ],
                "feat_strides": [
                    8,
                    16,
                    32
                ],
                "hidden_dim": 256,
                "use_encoder_idx": [
                    2
                ],
                "num_encoder_layers": 1,
                "nhead": 8,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "enc_act": "gelu",
                "expansion": 1.25,
                "depth_mult": 1.37,
                "act": "silu",
                "version": "deim",
                "csp_type": "csp2",
                "fuse_op": "sum"
            },
            "DEIMTransformer": {
                "feat_channels": [
                    256,
                    256,
                    256
                ],
                "feat_strides": [
                    8,
                    16,
                    32
                ],
                "hidden_dim": 256,
                "num_levels": 3,
                "num_layers": 6,
                "eval_idx": -1,
                "num_queries": 300,
                "num_denoising": 100,
                "label_noise_ratio": 0.5,
                "box_noise_scale": 1.0,
                "reg_max": 32,
                "reg_scale": 4,
                "layer_scale": 1,
                "num_points": [
                    3,
                    6,
                    3
                ],
                "cross_attn_method": "default",
                "query_select_method": "default",
                "activation": "silu",
                "mlp_act": "silu",
                "dim_feedforward": 2048,
                "eval_spatial_size": [
                    640,
                    640
                ]
            },
            "PostProcessor": {
                "num_top_queries": 300
            }
        }

        deimv2_x = DEIMv2(deimv2_x_config)
        grounding_model = DEIMv2.from_pretrained("Intellindust/DEIMv2_DINOv3_X_COCO")

    return grounding_model, processor, device


random.seed(233)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--country', type=str, default=None)
    parser.add_argument('--det_ckpt', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default="mask")
    args = parser.parse_args()
    return args


args = parse_args()

"""
python scenario_detection.py \
    --data_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/scenarios.json \
    --image_dir /weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo/train \
    --output_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/us_scenario.json \
    --country us \
    --det_ckpt deim 
    
python scenario_detection.py \
    --data_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/scenarios.json \
    --image_dir /weka/home/ext-yingzima/scratchaszalay1_ssci/yy/workshop/lingoqa/train \
    --output_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/uk_scenario.json \
    --country uk \
    --det_ckpt deim 
    
python scenario_detection.py \
    --data_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/scenarios.json \
    --image_dir /weka/home/ext-yingzima/scratchaszalay1_ssci/yy/workshop/once/data \
    --output_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/cn_scenario.json \
    --country cn \
    --det_ckpt deim 
    
python scenario_detection.py \
    --data_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/scenarios.json \
    --image_dir /weka/home/ext-yingzima/scratchaszalay1_ssci/yy/workshop/idd_multimodal/primary \
    --output_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/ind_scenario.json \
    --country ind \
    --det_ckpt deim 
    
python scenario_detection.py \
    --data_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/scenarios.json \
    --image_dir /weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/nuscenes/samples \
    --output_file /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/sg_scenario.json \
    --country sg \
    --det_ckpt deim 
"""

if __name__ == "__main__":
    PROMPT_TYPE_FOR_VIDEO = args.prompt_type
    grounding_model, processor, device = build_model(grounding_model_name=args.det_ckpt)


    with open(args.data_file, "rb") as f:
        data = json.load(f)

    from collections import defaultdict
    scenario2image = defaultdict(list)

    from tqdm import tqdm
    for scenario in tqdm(data, desc="Scenario"):
        object_name = scenario["keyword"]
        scenario2image[scenario["scenario_id"]] = []

        if args.country == "sg":
            with open("/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/CultureDrive/sg_scenario.json", "r") as f:
                data = json.load(f)

            for line in tqdm(data, desc="Image"):
                image_path = line['image_path'][-1]
                image = Image.open(image_path)

                boxes, labels = [], []
                if args.det_ckpt == "dino":
                    input_boxes, input_labels, input_scores = dino_detect_object(image, object_name, grounding_model,
                                                                                 processor)
                    remove_ids = [idx for idx, label in enumerate(labels) if label + "." not in object_name]
                    boxes = [item for idx, item in enumerate(boxes) if idx not in remove_ids]
                    labels = [item for idx, item in enumerate(labels) if idx not in remove_ids]
                    boxes, labels = filter_overlap_keep_smaller(boxes, labels)

                elif args.det_ckpt == "deim":
                    input_boxes, input_labels, input_scores = deim_detect_object(image, object_name, grounding_model,
                                                                                 processor)

                boxes.extend(input_boxes)
                labels.extend(input_labels)

                scenario2image[scenario["scenario_id"]].append(
                    {
                        "image_path": line['image_path'],
                        "object_info": {
                            "labels": labels,
                            "boxes": boxes,
                        },
                        "gt_planning": line['gt_planning'],
                        "gt_planning_mask": line['gt_planning_mask'],
                        "gt_planning_command": line['gt_planning_command'],
                    }
                )

        else:
            for scene_id in tqdm(os.listdir(args.image_dir), desc="Image"):
                scene_path = os.path.join(args.image_dir, scene_id)
                if args.country == "cn":
                    scene_path = os.path.join(scene_path, "cam01")

                elif args.country == "ind":
                    scene_path = os.path.join(scene_path, "leftCamImgs")


                for i, image_id in enumerate(os.listdir(scene_path)):

                    # waymo
                    if args.country == "us":
                        if i % 5 != 0: continue
                        if "JOINT" not in image_id: continue

                    # lingoqa
                    elif args.country == "uk":
                        if "jpg" not in image_id: continue

                    # once and driveaction
                    elif args.country == "cn":
                        if "jpg" not in image_id: continue
                        # print(int(image_id.split(".")[0]))
                        if int(image_id.split(".")[0]) % 2000 != 0:
                            continue
                        # print(int(image_id.split(".")[0]))
                    # idd
                    elif args.country == "ind":
                        if "jpg" not in image_id: continue

                    image_path = os.path.join(scene_path, image_id)  #
                    image = Image.open(image_path)


                    boxes, labels = [], []
                    if args.det_ckpt == "dino":
                        input_boxes, input_labels, input_scores = dino_detect_object(image, object_name, grounding_model, processor)
                        remove_ids = [idx for idx, label in enumerate(labels) if label + "." not in object_name]
                        boxes = [item for idx, item in enumerate(boxes) if idx not in remove_ids]
                        labels = [item for idx, item in enumerate(labels) if idx not in remove_ids]
                        boxes, labels = filter_overlap_keep_smaller(boxes, labels)

                    elif args.det_ckpt == "deim":
                        input_boxes, input_labels, input_scores = deim_detect_object(image, object_name, grounding_model,
                                                                                     processor)

                    boxes.extend(input_boxes)
                    labels.extend(input_labels)


                    scenario2image[scenario["scenario_id"]].append(
                        {
                            "image_path": image_path,
                            "object_info": {
                                "labels": labels,
                                "boxes": boxes,
                            }
                        }
                    )

                # print(scenario2image)

        break



    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(scenario2image, f, ensure_ascii=False, indent=2)

