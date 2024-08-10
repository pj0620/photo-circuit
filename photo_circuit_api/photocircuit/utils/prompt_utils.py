import base64
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_prompt(prompt: str) -> str:
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  with open(f'{current_directory}/../prompts/{prompt}') as f:
    return f.read()
  
  
def generate_image_with_grid_base64(input_base64: str, step_size: int, include_grid: bool = True) -> str:
  """
  Decodes a base64 encoded PNG, generates an image with a grid overlay,
  and returns it as a base64 encoded PNG.

  Args:
  input_base64 (str): Base64 encoded PNG image.

  Returns:
  str: Base64 encoded PNG of the image with grid overlay.
  """
  
  # Decode the base64 input to get the image
  image_data = base64.b64decode(input_base64)
  image = Image.open(BytesIO(image_data))
  image_array = np.array(image)
  
  image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
  
  # Create a figure and axis
  fig, ax = plt.subplots()
  
  # Display the image
  ax.imshow(image_array)
  
  x_step_size = step_size
  y_step_size = step_size
  
  # Set the major ticks at intervals of 10 pixels
  ax.set_xticks(np.arange(0, image_array.shape[1], x_step_size))
  ax.set_yticks(np.arange(0, image_array.shape[0], y_step_size))
  
  # Grid lines based on major ticks
  if include_grid:
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
  
  # Optionally add labels to major ticks
  ax.set_xticklabels(np.arange(0, image_array.shape[1], x_step_size))
  ax.set_yticklabels(np.arange(0, image_array.shape[0], y_step_size))
  
  # Save the plot to a BytesIO object in PNG format
  buf = BytesIO()
  plt.savefig(buf, format='png')
  plt.close(fig)  # Close the figure to free up memory
  # Encode the PNG image to base64
  buf.seek(0)
  output_base64 = base64.b64encode(buf.read()).decode('utf-8')
  return output_base64

