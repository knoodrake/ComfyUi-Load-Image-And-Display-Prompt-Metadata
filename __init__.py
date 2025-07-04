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
                "negative_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive_prompt", "negative_prompt")

    CATEGORY = "testt"
    FUNCTION = "load_image"

    def load_image(self, image, positive_prompt="", negative_prompt=""):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

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
        
        return (output_image, output_mask, positive_prompt, negative_prompt)

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