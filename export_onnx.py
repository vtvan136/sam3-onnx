#!/usr/bin/env python3

import pathlib
import numpy as np
import onnxruntime
import PIL.Image
import torch
import imgviz
import gc
from loguru import logger
from osam._models.yoloworld.clip import tokenize

from infer_torch import get_replace_freqs_cis
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# Cấu hình thiết bị
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. WRAPPERS (IMAGE, LANGUAGE, DECODER)
# ==========================================
class _ImageEncoder(torch.nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std
        out = self.backbone._forward_image_no_act_ckpt(x)
        return (
            out["vision_pos_enc"][0], out["vision_pos_enc"][1], out["vision_pos_enc"][2],
            out["backbone_fpn"][0], out["backbone_fpn"][1], out["backbone_fpn"][2]
        )

class _LanguageEncoder(torch.nn.Module):
    def __init__(self, language_model) -> None:
        super().__init__()
        self.lm = language_model

    def forward(self, tokens: torch.Tensor):
        text_attention_mask = (tokens != 0).bool()
        _, text_memory = self.lm.encoder(tokens)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.lm.resizer(text_memory)
        inputs_embeds = self.lm.encoder.token_embedding(tokens)
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)

class _Decoder(torch.nn.Module):
    def __init__(self, model, processor) -> None:
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, oh, ow, vpe0, vpe1, vpe2, bfp0, bfp1, bfp2, l_mask, l_feat, l_emb, b_coords, b_labels, b_masks):
        b_coords = b_coords.to(dtype=bfp0.dtype)
        geom_prompt = self.model._get_dummy_prompt()
        geom_prompt.box_embeddings = b_coords
        geom_prompt.box_labels = b_labels.to(torch.int64)
        geom_prompt.box_mask = b_masks.to(dtype=bfp0.dtype)
        
        state = {
            "original_height": oh.to(torch.int64),
            "original_width": ow.to(torch.int64),
            "backbone_out": {
                "vision_pos_enc": [vpe0, vpe1, vpe2],
                "backbone_fpn": [bfp0, bfp1, bfp2],
                "language_mask": l_mask.bool(),
                "language_features": l_feat,
                "language_embeds": l_emb,
            },
            "geometric_prompt": geom_prompt,
        }
        res = self.processor._forward_grounding(state)
        return res["boxes"], res["scores"], res["masks"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_session(path, use_gpu=True):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if use_gpu:
        providers = [('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    return onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 3. EXPORT & RUN LOGIC
# ==========================================

def run_image_encoder(processor, image):
    onnx_path = "models/sam3_image_encoder_fp16.onnx"
    if not pathlib.Path(onnx_path).exists():
        logger.info("Exporting Image Encoder (FP16)...")
        model = _ImageEncoder(processor.model.backbone).to(DEVICE).half().eval()
        dummy_in = torch.randn(1, 3, 1008, 1008).to(DEVICE).half()
        torch.onnx.export(model, (dummy_in,), onnx_path, opset_version=17)
        del model
        clear_gpu()

    # CHẠY TRÊN CPU ĐỂ TRÁNH LỖI SOFTMAX OOM
    sess = get_session(onnx_path, use_gpu=False)
    img_input = (np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1) / 255.0).astype(np.float16)
    out = sess.run(None, {"image": img_input[None, :]})
    return out[:3], out[3:]

def run_language_encoder(processor):
    onnx_path = "models/sam3_language_encoder_fp16.onnx"
    if not pathlib.Path(onnx_path).exists():
        logger.info("Exporting Language Encoder...")
        lm = _LanguageEncoder(processor.model.backbone.language_backbone).to(DEVICE).half().eval()
        tokens = torch.from_numpy(tokenize(texts=["person"], context_length=32)).to(DEVICE)
        torch.onnx.export(lm, (tokens,), onnx_path, opset_version=17)
        del lm
        clear_gpu()

    sess = get_session(onnx_path, use_gpu=True)
    tokens = tokenize(texts=["person"], context_length=32).astype(np.int64)
    return sess.run(None, {"tokens": tokens})

def run_decoder(original_size, v_pos, b_fpn, l_out, boxes_np):
    onnx_path = "models/sam3_decoder_fp16.onnx"
    if not pathlib.Path(onnx_path).exists():
        logger.info("Exporting Decoder...")
        full_model = build_sam3_image_model().to(DEVICE).half().eval()
        processor = Sam3Processor(full_model)
        decoder = _Decoder(full_model, processor).to(DEVICE).half().eval()
        
        dummy_args = (
            torch.tensor([original_size[0]], dtype=torch.float16, device=DEVICE),
            torch.tensor([original_size[1]], dtype=torch.float16, device=DEVICE),
            *[torch.from_numpy(x).to(DEVICE).half() for x in v_pos],
            *[torch.from_numpy(x).to(DEVICE).half() for x in b_fpn],
            torch.from_numpy(l_out[0]).to(DEVICE).bool(),
            torch.from_numpy(l_out[1]).to(DEVICE).half(),
            torch.from_numpy(l_out[2]).to(DEVICE).half(),
            torch.from_numpy(boxes_np['coords']).to(DEVICE).half(),
            torch.from_numpy(boxes_np['labels']).to(DEVICE).long(),
            torch.from_numpy(boxes_np['masks'].astype(np.float32)).to(DEVICE).half(),
        )

        torch.onnx.export(decoder, dummy_args, onnx_path, opset_version=17, do_constant_folding=True)
        del full_model, processor, decoder
        clear_gpu()

    sess = get_session(onnx_path, use_gpu=True)
    inputs = {
        "oh": np.array([original_size[0]], dtype=np.float16),
        "ow": np.array([original_size[1]], dtype=np.float16),
        "vpe0": v_pos[0], "vpe1": v_pos[1], "vpe2": v_pos[2],
        "bfp0": b_fpn[0], "bfp1": b_fpn[1], "bfp2": b_fpn[2],
        "l_mask": l_out[0].astype(bool), "l_feat": l_out[1], "l_emb": l_out[2],
        "b_c": boxes_np['coords'].astype(np.float16),
        "b_l": boxes_np['labels'].astype(np.int64),
        "b_m": boxes_np['masks'].astype(np.float16),
    }
    return sess.run(None, inputs)

# ==========================================
# 4. MAIN
# ==========================================
def main():
    pathlib.Path("models").mkdir(exist_ok=True)
    
    # Khởi tạo model PyTorch tạm thời để export
    raw_model = build_sam3_image_model().to(DEVICE).half().eval()
    get_replace_freqs_cis(raw_model)
    processor = Sam3Processor(raw_model)

    image = PIL.Image.open("images/bus.jpg")
    
    # Bước 1 & 2: Chạy Encoders
    v_pos, b_fpn = run_image_encoder(processor, image)
    l_out = run_language_encoder(processor)

    # Bước 3: Prompt (Box)
    boxes_data = {
        'coords': np.array([[[100, 100, 400, 800]]], dtype=np.float32),
        'labels': np.array([[1]], dtype=np.int64),
        'masks': np.array([[True]], dtype=np.bool_)
    }
    
    # Bước 4: Chạy Decoder (Sử dụng kết quả từ Encoders)
    logger.info("Running Decoder Inference...")
    res_boxes, res_scores, res_masks = run_decoder(
        (image.height, image.width), v_pos, b_fpn, l_out, boxes_data
    )

    logger.success("Inference successful!")
    
    # Hiển thị kết quả
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=res_masks[0].astype(bool),
        bboxes=res_boxes[0],
        labels=np.arange(len(res_boxes[0])) + 1,
        captions=[f"{s:.2f}" for s in res_scores[0]],
    )
    imgviz.io.pil_imshow(viz)

if __name__ == "__main__":
    main()