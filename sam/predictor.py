import numpy as np

from sam.vit import Vit
from sam.decoder import Decoder


class SamPredictor:
    def __init__(self, vit_model_path, decoder_model_path, device="cuda", warmup_epoch=5, **kwargs):
        self.vit = Vit(vit_model_path, device, warmup_epoch, **kwargs)
        self.decoder = Decoder(decoder_model_path, device, warmup_epoch, **kwargs)

        self.features = None
        self.origin_image_size = None

    def register_image(self, img):
        self.origin_image_size = img.shape
        self.features = self.vit.run(img)

    def get_mask(self,
                 point_coords,
                 point_labels,
                 boxes=None,
                 mask_input=None
                 ):
        result = self.decoder.run(self.features,
                                  self.origin_image_size[:2],
                                  point_coords,
                                  point_labels,
                                  boxes,
                                  mask_input)
        return result
