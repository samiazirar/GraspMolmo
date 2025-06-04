import re
import xml.etree.ElementTree as ElementTree
from typing import Optional

import numpy as np
from torch import Tensor
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from graspmolmo.inference.utils import get_grasp_points, draw_grasp_points, draw_grasp


def parse_point(pred: str, image_size: Optional[tuple[int, int]] = None):
    """
    Args:
        pred: The prediction string from the model.
        image_size: The size of the image, (width, height). If provided, return in pixels, otherwise return in normalized coordinates.
    Returns:
        The predicted point as a numpy array of shape (2,).
    """
    point_xmls = re.findall(r'<points?.*?</points?>', pred, re.DOTALL)
    if len(point_xmls) == 0:
        print(f"Invalid prediction: {pred}")
        return None
    point_xml = point_xmls[0]
    try:
        point_elem = ElementTree.fromstring(point_xml)
        
        if point_elem is not None:
            if point_elem.tag == 'point':
                x = float(point_elem.get('x'))
                y = float(point_elem.get('y'))
            elif point_elem.tag == 'points':
                x = float(point_elem.get('x1'))
                y = float(point_elem.get('y1'))
            else:
                print(f"Invalid prediction: {pred}")
                return None
            ret = np.array([x, y])
            if image_size is not None:
                ret = ret / 100 * np.array(image_size)
            return ret
        else:
            print("No point element found in XML")
    except ElementTree.ParseError as e:
        print(f"Failed to parse XML: {e}")
    return None

class GraspMolmo:
    def __init__(self):
        self.model_name = "allenai/GraspMolmo"
        self.prompt_pfx = "Point to the grasp that would accomplish the following task: "

        self.processor = AutoProcessor.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        self.gen_cfg = GenerationConfig(max_new_tokens=256, stop_strings="<|endoftext|>")

    def _pred(self, image: Image.Image, task: str, verbosity: int = 0) -> str:
        inputs: dict[str, Tensor] = self.processor.process(
            text=f"{self.prompt_pfx}{task}",
            images=[image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output = self.model.generate_from_batch(inputs, self.gen_cfg, tokenizer=self.processor.tokenizer)
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if verbosity >= 1:
            print("Output:", generated_text)
        return generated_text

    def pred_points(self, image: Image.Image, task: str, verbosity: int = 0):
        """
        Args:
            image: The image of the scene.
            task: The task to predict the grasp point for.
            verbosity: The verbosity level, higher is more.
        Returns:
            The predicted point as a numpy array of shape (2,).
        """
        pred = self._pred(image, task, verbosity)
        point = parse_point(pred, image.size)

        if verbosity >= 1:
            print(f"Predicted point: {point}")

        if verbosity >= 3 and point is not None:
            draw = ImageDraw.Draw(image)
            r = 5
            draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill="blue")

        return point

    def pred_grasp(self, image: Image.Image, pc: np.ndarray, task: str, grasps: np.ndarray, cam_K: np.ndarray, verbosity: int = 0):
        """
        Args:
            image: The image of the scene.
            pc: (*, 3) The point cloud of the scene.
            task: The task to perform.
            grasps: (N, 4, 4) The grasps to choose from, in camera frame.
            cam_K: (3, 3) The camera intrinsic matrix.
        Returns:
            The index of the grasp to perform.
        """
        point = self.pred_points(image, task, verbosity=verbosity)
        if point is None:
            return None

        grasp_points = get_grasp_points(pc, grasps)
        grasp_points_2d = grasp_points @ cam_K.T
        grasp_points_2d = grasp_points_2d[:, :2] / grasp_points_2d[:, 2:3]

        dists = np.linalg.norm(grasp_points_2d - point[None], axis=1)
        grasp_idx = np.argmin(dists).item()

        if verbosity >= 4:
            draw_grasp_points(image, cam_K, pc, grasps, r=5, color="red")
            draw_grasp(image, cam_K, grasps[grasp_idx], color="blue")

        return grasp_idx
