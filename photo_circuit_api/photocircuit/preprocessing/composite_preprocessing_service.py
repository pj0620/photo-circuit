import numpy as np

from photocircuit.preprocessing.base_preprocessing_service import BasePreprocessingService


class CompositePreprocessingService(BasePreprocessingService):
  def __init__(self, *chain: BasePreprocessingService):
    self.chain = chain
    
  def preprocess_image(self, image_arr: np.array) -> np.array:
    cur: np.array = image_arr
    for step in self.chain:
      cur = step.preprocess_image(cur)
    return cur
    
  