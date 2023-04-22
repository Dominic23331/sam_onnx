from typing import Union

import onnxruntime as ort
import numpy as np
from tqdm import tqdm

from sam.transforms import apply_coords, apply_boxes


class Decoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        model_path (str): decoder model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 10.
    """
    img_size = (1024, 1024)
    mask_threshold = 0.0

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 warmup_epoch: int = 10,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading decoder model...")
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch: int) -> None:
        """warmup function

        Args:
            epoch (int): warmup epoch.
        """
        x = {"image_embeddings": np.random.random((1, 256, 64, 64)).astype(np.float32),
             "point_coords": np.random.random((1, 1, 2)).astype(np.float32),
             "point_labels": np.ones((1, 1), dtype=np.float32),
             "mask_input": np.random.random((1, 1, 256, 256)).astype(np.float32),
             "has_mask_input": np.ones((1,), dtype=np.float32),
             "orig_im_size": np.array((1024, 1024), dtype=np.float32)}
        print("start warmup!")
        for i in tqdm(range(epoch)):
            self.session.run(None, x)
        print("warmup finish!")

    def run(self,
            img_embeddings: np.ndarray,
            origin_image_size: Union[list, tuple],
            point_coords: Union[list, np.ndarray] = None,
            point_labels: Union[list, np.ndarray] = None,
            boxes: Union[list, np.ndarray] = None,
            mask_input: np.ndarray = None) -> dict:
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=256.

        Returns:
            dict: the segment results.
        """
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = apply_coords(point_coords, origin_image_size, self.img_size[0]).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = apply_boxes(boxes, origin_image_size, self.img_size[0]).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {"image_embeddings": img_embeddings,
                      "point_coords": point_coords,
                      "point_labels": point_labels,
                      "mask_input": mask_input,
                      "has_mask_input": has_mask_input,
                      "orig_im_size": np.array(origin_image_size, dtype=np.float32)}
        res = self.session.run(None, input_dict)

        result_dict = dict()
        for i in range(len(res)):
            out_name = self.session.get_outputs()[i].name
            if out_name == "masks":
                mask = (res[i] > self.mask_threshold).astype(np.int32)
                result_dict[out_name] = mask
            else:
                result_dict[out_name] = res[i]

        return result_dict
