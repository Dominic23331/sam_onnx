import onnxruntime as ort
import numpy as np
from tqdm import tqdm

from sam.transforms import apply_coords, apply_boxes


class Decoder:
    img_size = (1024, 1024)
    mask_threshold = 0.0

    def __init__(self, model_path, device="cuda", warmup_epoch=10, **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch):
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
            img_embeddings,
            origin_image_size,
            point_coords,
            point_labels,
            boxes=None,
            mask_input=None):
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
        if isinstance(point_coords, list):
            point_coords = np.array(point_coords, dtype=np.float32)
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=np.float32)

        point_coords = apply_coords(point_coords, origin_image_size, self.img_size[0]).astype(np.float32)
        point_coords = np.expand_dims(point_coords, axis=0)
        point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = apply_boxes(boxes, origin_image_size, self.img_size[0]).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            point_coords = np.concatenate([point_coords, boxes], axis=1)
            point_labels = np.concatenate([point_labels, box_label], axis=1)

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
                mask = res[i] > self.mask_threshold
                result_dict[out_name] = mask
            else:
                result_dict[out_name] = res[i]

        return result_dict
