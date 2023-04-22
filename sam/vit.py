import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm


class Vit:
    """Vit encoder model for sam.

    In this class, vit model will encoder the input image.

    Args:
        model_path (str): Vit model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 5.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 warmup_epoch: int = 5,
                 **kwargs):
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

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch: int) -> None:
        """warmup function

        Args:
            epoch (int): warmup epoch.
        """
        x = np.random.random(self.input_shape).astype(np.float32)
        print("start warmup!")
        for i in tqdm(range(epoch)):
            self.session.run(None, {self.input_name: x})
        print("warmup finish!")

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use vit to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image.

        Returns:
            np.ndarray: image`s feature.
        """
        assert list(tensor.shape) == self.input_shape
        feature = self.session.run(None, {self.input_name: tensor})[0]
        assert list(feature.shape) == self.output_shape
        return feature

    def transform(self, img: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for vit.

        Args:
            img (np.ndarray): input image, the image type should be BGR.

        Returns:
            np.ndarray: transformed image.
        """
        h, w, c = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std

        size = max(h, w)
        img = np.pad(img, ((0, size - h), (0, size - w), (0, 0)), 'constant', constant_values=(0, 0))
        img = cv2.resize(img, self.input_shape[2:])
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, axes=[0, 3, 1, 2]).astype(np.float32)
        return img

    def run(self, img: np.ndarray) -> np.ndarray:
        """Vit forward function

        This function can transform the input image and
        use vit to extract feature from input image.

        Returns:
            np.ndarray: the input image`s feature.
        """
        tensor = self.transform(img)
        features = self._extract_feature(tensor)
        return features
