import base64
import io

import cv2
import numpy as np
from PIL import Image


def base64_to_numpy(base64_str: str) -> np.ndarray:
  img_data = base64.b64decode(base64_str)
  buffer = io.BytesIO(img_data)
  img = Image.open(buffer)
  array = np.array(img)
  return array


def numpy_to_base64(array: np.ndarray) -> str:
  # Convert the NumPy array to a PIL Image
  image = Image.fromarray(array.astype('uint8'))
  
  # Create a bytes buffer for the image
  buffer = io.BytesIO()
  
  # Save the image to the buffer in PNG format
  image.save(buffer, format='PNG')
  
  # Get the byte data of the image
  byte_data = buffer.getvalue()
  
  # Encode the byte data to a base64 string
  base64_str = base64.b64encode(byte_data).decode('utf-8')
  
  return base64_str


def scale_image(image: np.array, screen_size: int) -> np.array:
  width, height = image.shape
  max_dim = max(width, height)
  scaling_factor = screen_size / max_dim
  return cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
