import cv2
import numpy as np

from photocircuit.preprocessing.base_preprocessing_service import BasePreprocessingService
from photocircuit.utils.common import scale_image

FIXED_SIZE = 600


class ScalingPreprocessingService(BasePreprocessingService):
  def preprocess_image(self, image_arr: np.array) -> np.array:
    max_side_len = max(image_arr.shape)
    scale_factor = (FIXED_SIZE * (1 + max_side_len // FIXED_SIZE)) / max_side_len
    scaled_circuit_img_arr = cv2.resize(
      image_arr,
      (0, 0),
      fx=scale_factor,
      fy=scale_factor,
      interpolation=cv2.INTER_LINEAR
    )
    return scaled_circuit_img_arr
