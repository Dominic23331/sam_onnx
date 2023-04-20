import numpy as np

from sam.vit import Vit
from sam.decoder import Decoder


class SamPredictor:
    def __init__(self, vit_model_path, decoder_model_path, device, warmup_epoch=10, **kwargs):
        self.vit = Vit(vit_model_path, device, warmup_epoch, **kwargs)
        self.decoder = Decoder(decoder_model_path, device, warmup_epoch, **kwargs)

        self.features = None

    def register_image(self, img):
        self.features = self.vit.run(img)
