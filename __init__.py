from __future__ import annotations
import torch

import os
import sys
import json
import hashlib

from PIL import Image, ImageOps, ImageSequence
import numpy as np

import comfy.model_management
import folder_paths
import node_helpers


def _extract_from_workflow(workflow: dict) -> dict:
    """Extract prompts and sampler parameters from a ComfyUI workflow."""
    meta = {"positive": "", "negative": "", "seed": None, "steps": None, "cfg": None}

    positive_id = None
    negative_id = None

    for node_id, node in workflow.items():
        if node.get("class_type") in ("KSampler", "KSamplerAdvanced") and node.get("inputs"):
            inputs = node["inputs"]
            if not positive_id and isinstance(inputs.get("positive"), list):
                positive_id = str(inputs["positive"][0])
            if not negative_id and isinstance(inputs.get("negative"), list):
                negative_id = str(inputs["negative"][0])

            if meta["seed"] is None and isinstance(inputs.get("seed"), (int, float)):
                meta["seed"] = int(inputs["seed"])
            if meta["steps"] is None and isinstance(inputs.get("steps"), (int, float)):
                meta["steps"] = int(inputs["steps"])
            if meta["cfg"] is None and isinstance(inputs.get("cfg"), (int, float)):
                meta["cfg"] = float(inputs["cfg"])

    if not positive_id and not negative_id:
        for node_id, node in workflow.items():
            if node.get("class_type") == "CFGGuider" and node.get("inputs"):
                inputs = node["inputs"]
                if not positive_id and isinstance(inputs.get("positive"), list):
                    pos_id = inputs["positive"][0]
                    pos_node = workflow.get(pos_id)
                    if pos_node and pos_node.get("class_type") == "ChromaPaddingRemoval" and pos_node.get("inputs") and pos_node["inputs"].get("conditioning"):
                        positive_id = str(pos_node["inputs"]["conditioning"][0])
                    else:
                        positive_id = str(pos_id)
                if not negative_id and isinstance(inputs.get("negative"), list):
                    neg_id = inputs["negative"][0]
                    neg_node = workflow.get(neg_id)
                    if neg_node and neg_node.get("class_type") == "ChromaPaddingRemoval" and neg_node.get("inputs") and neg_node["inputs"].get("conditioning"):
                        negative_id = str(neg_node["inputs"]["conditioning"][0])
                    else:
                        negative_id = str(neg_id)
            if node.get("class_type") == "NAGCFGGuider" and node.get("inputs"):
                inputs = node["inputs"]
                if not positive_id and isinstance(inputs.get("positive"), list):
                    positive_id = str(inputs["positive"][0])
                if not negative_id and isinstance(inputs.get("nag_negative"), list):
                    negative_id = str(inputs["nag_negative"][0])

    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode" and node.get("inputs") and node["inputs"].get("text") is not None:
            text = node["inputs"]["text"]
            if positive_id and str(node_id) == positive_id:
                meta["positive"] = text
            elif negative_id and str(node_id) == negative_id:
                meta["negative"] = text
            else:
                title = (node.get("_meta", {}).get("title", "").lower())
                if "negative" in title or "nag" in title:
                    if not meta["negative"]:
                        meta["negative"] = text
                else:
                    if not meta["positive"]:
                        meta["positive"] = text
        elif node.get("class_type") == "CLIPTextEncodeFlux" and node.get("inputs"):
            text = node["inputs"].get("clip_l") or node["inputs"].get("t5xxl") or ""
            if positive_id and str(node_id) == positive_id:
                meta["positive"] = text
            elif negative_id and str(node_id) == negative_id:
                meta["negative"] = text

    return meta

class LoadImageX:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 0}),
                "cfg": ("FLOAT", {"default": 0.0})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "MASK", "positive_prompt", "negative_prompt", "seed", "steps", "cfg")

    CATEGORY = "testt"
    FUNCTION = "load_image"

    def load_image(self, image, positive_prompt="", negative_prompt="", seed=0, steps=0, cfg=0.0):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        # Attempt to extract metadata from the image if available
        meta_prompt = img.info.get("prompt")
        if meta_prompt:
            try:
                workflow = json.loads(meta_prompt.replace(": NaN", ": null"))
                meta = _extract_from_workflow(workflow)
                if meta.get("positive"):
                    positive_prompt = meta["positive"]
                if meta.get("negative"):
                    negative_prompt = meta["negative"]
                if meta.get("seed") is not None:
                    seed = int(meta["seed"])
                if meta.get("steps") is not None:
                    steps = int(meta["steps"])
                if meta.get("cfg") is not None:
                    cfg = float(meta["cfg"])
            except Exception:
                pass

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image_rgb = i.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask_np)

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask, positive_prompt, negative_prompt, seed, steps, cfg)

    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

NODE_CLASS_MAPPINGS = { "LoadImageX": LoadImageX }
NODE_DISPLAY_NAME_MAPPINGS = { "LoadImageX": "Load Image And Display Prompt Metadata" }
WEB_DIRECTORY = "./web"
