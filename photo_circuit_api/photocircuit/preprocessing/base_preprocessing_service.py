from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePreprocessingService(ABC):
  @abstractmethod
  def preprocess_image(self, image_arr: np.array) -> np.array:
    """
    Simply scales the image to a preset size
    
    :param image_arr: raw circuit image from user as numpy array
    :return: image after preprocessing
    """
    
    
