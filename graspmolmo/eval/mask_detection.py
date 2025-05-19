import urllib.request
import os

from PIL import Image
import torch
import numpy as np

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2, HF_MODEL_ID_TO_FILENAMES
from sam2.sam2_image_predictor import SAM2ImagePredictor

from graspmolmo.eval.utils import download

class MaskDetector:
    def __init__(self, dino_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="sam2.1_hiera_base_plus", device="cuda"):
        self.device = device
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{sam_model_id}.pt"
        model_path = download(url, f"{sam_model_id}.pt")

        cfg_name = HF_MODEL_ID_TO_FILENAMES[f"facebook/{sam_model_id}".replace("_", "-")][0]
        sam2_model = build_sam2(cfg_name, model_path, device=device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)

    def to(self, device: str):
        self.device = device
        self.dino_model.to(device)
        self.sam_predictor.to(device)

    @torch.no_grad()
    def detect_bbox(self, object_name: str, image: Image.Image) -> np.ndarray:
        text = f"a {object_name.lower()}."
        dino_inputs = self.dino_processor(images=image, text=text, return_tensors="pt").to(self.device)
        dino_outputs = self.dino_model(**dino_inputs)
        dino_results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in dino_results[0].items()}
        if len(result["scores"]) == 0:
            return None
        best_idx = np.argmax(result["scores"])
        bbox = result["boxes"][best_idx]
        return bbox

    @torch.no_grad()
    def detect_mask(self, object_name: str, image: Image.Image) -> np.ndarray:
        bbox = self.detect_bbox(object_name, image)
        if bbox is None:
            return None

        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            self.sam_predictor.set_image(image)
            masks, scores, _ = self.sam_predictor.predict(box=bbox[None])

        mask_idx = np.argmax(scores)
        mask = masks[mask_idx].astype(bool)
        return mask
