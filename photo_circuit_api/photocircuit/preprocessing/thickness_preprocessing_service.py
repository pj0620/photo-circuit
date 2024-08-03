import cv2
import numpy as np

from photocircuit.preprocessing.base_preprocessing_service import BasePreprocessingService
from photocircuit.utils.common import scale_image


LINE_THICKNESS = 4


class ThicknessPreprocessingService(BasePreprocessingService):
  def preprocess_image(self, image_arr: np.array) -> np.array:
    # Check the number of channels in the image
    if len(image_arr.shape) == 2 or image_arr.shape[2] == 1:  # Grayscale image
      gray_image = image_arr
    else:  # RGB image
      gray_image = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to get binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    # Find edges using Canny
    edges = cv2.Canny(binary_image, 50, 150)
    
    # Dilate the edges to make lines thicker
    kernel = np.ones((LINE_THICKNESS, LINE_THICKNESS), np.uint8)
    thickened_image = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert the thickened image back to a 3-channel image
    thickened_image = cv2.cvtColor(thickened_image, cv2.COLOR_GRAY2RGB)
    
    return thickened_image
