#!/usr/bin/env python3
# ==========================================
# SAM3 ONNX Inference
# Select device: auto / cpu / gpu
# ==========================================

import argparse
import pathlib
import sys

import cv2
import imgviz
import numpy as np
import PIL.Image
from loguru import logger
from osam._models.yoloworld.clip import tokenize

import onnxruntime as ort


# ==========================================
# SESSION CREATOR
# ==========================================
def create_session(model_path: str, device: str) -> ort.InferenceSession:
    available = ort.get_available_providers()

    if device == "cpu":
        logger.info("Forcing CPUExecutionProvider")
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    if device == "gpu":
        if "CUDAExecutionProvider" not in available:
            logger.error("CUDAExecutionProvider not available!")
            sys.exit(1)

        logger.info("Forcing CUDAExecutionProvider")
        return ort.InferenceSession(
            model_path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ],
        )

    # AUTO MODE
    if device == "auto":
        if "CUDAExecutionProvider" in available:
            logger.info("Auto mode → Using GPU")
            return ort.InferenceSession(
                model_path,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
            )

        logger.info("Auto mode → Using CPU")
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    raise ValueError("Invalid device option")


# ==========================================
# CLI
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=pathlib.Path, required=True)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Select inference device",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text-prompt", type=str)
    group.add_argument("--box-prompt", type=str, nargs="?", const="0,0,0,0")

    args = parser.parse_args()

    if args.box_prompt:
        args.box_prompt = [float(x) for x in args.box_prompt.split(",")]
        if len(args.box_prompt) != 4:
            logger.error("box_prompt must be cx,cy,w,h")
            sys.exit(1)

    return args


# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_args()

    logger.info("Selected device: {}", args.device)

    # ---- Create Sessions ----
    sess_image = create_session("models/sam3_image_encoder.onnx", args.device)
    sess_language = create_session("models/sam3_language_encoder.onnx", args.device)
    sess_decode = create_session("models/sam3_decoder.onnx", args.device)

    logger.info("Image providers: {}", sess_image.get_providers())
    logger.info("Language providers: {}", sess_language.get_providers())
    logger.info("Decoder providers: {}", sess_decode.get_providers())

    # ---- Load Image ----
    image = PIL.Image.open(args.image).convert("RGB")
    original_w, original_h = image.size

    image_input = (
        np.asarray(image.resize((1008, 1008)), dtype=np.uint8)
        .transpose(2, 0, 1)
    )

    # =====================
    # IMAGE ENCODER
    # =====================
    logger.info("Running image encoder...")
    output = sess_image.run(None, {"image": image_input})
    vision_pos_enc = output[:3]
    backbone_fpn = output[3:]

    # =====================
    # LANGUAGE ENCODER
    # =====================
    text_prompt = args.text_prompt if args.text_prompt else "visual"

    logger.info("Running language encoder...")
    output = sess_language.run(
        None,
        {"tokens": tokenize([text_prompt], context_length=32)},
    )
    language_mask = output[0]
    language_features = output[1]

    # =====================
    # DECODER
    # =====================
    logger.info("Running decoder...")

    box_coords = np.array(
        args.box_prompt if args.box_prompt else [0, 0, 0, 0],
        dtype=np.float32,
    ).reshape(1, 1, 4)

    box_labels = np.array([[1]], dtype=np.int64)
    box_masks = np.array([[False]], dtype=np.bool_)

    boxes, scores, masks = sess_decode.run(
        None,
        {
            "original_height": np.array(original_h, dtype=np.int64),
            "original_width": np.array(original_w, dtype=np.int64),
            "backbone_fpn_0": backbone_fpn[0],
            "backbone_fpn_1": backbone_fpn[1],
            "backbone_fpn_2": backbone_fpn[2],
            "vision_pos_enc_2": vision_pos_enc[2],
            "language_mask": language_mask,
            "language_features": language_features,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        },
    )

    # =====================
    # VISUALIZATION
    # =====================
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
        font_size=max(1, min(image.size) // 40),
    )

    PIL.Image.fromarray(viz).show()


if __name__ == "__main__":
    main()
