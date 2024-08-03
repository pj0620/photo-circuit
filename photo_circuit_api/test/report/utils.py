import base64
import io
import os
from io import BytesIO

from PIL import Image
from matplotlib import pyplot as plt

from photocircuit.component_detection.model import ComponentName, Component, CircuitComponents


def get_image_path_for_comp(component_name: ComponentName):
  file_path = os.path.abspath(__file__)
  test_data_dir = os.path.dirname(file_path) + "/../report/images"
  image_map = {
    ComponentName.RESISTOR: 'resistor.png',
    ComponentName.CAPACITOR: 'capacitor.png',
    ComponentName.VOLTAGE_SOURCE: 'voltage_source.png',
    ComponentName.CURRENT_SOURCE: 'current_source.png',
    ComponentName.INDUCTOR: 'inductor.png',
    ComponentName.DEPENDANT_VOLTAGE_SOURCE: 'dependant_voltage_source.png',
    ComponentName.DEPENDANT_CURRENT_SOURCE: 'dependant_current_source.png',
    ComponentName.UNKNOWN: 'unknown.png'
  }
  comp_filename = image_map.get(component_name, f'unknown.png')
  return f'{test_data_dir}/{comp_filename}'


def get_generated_circuit(circuit_components: CircuitComponents):
  canvas_width = max(c.position.x for c in circuit_components.components)
  canvas_height = max(c.position.y for c in circuit_components.components)
  
  canvas_width = int(canvas_width + 100)
  canvas_height = int(canvas_height + 100)
  
  # Create a blank canvas
  final_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
  
  for component in circuit_components.components:
    # Load the component image
    component_image = Image.open(get_image_path_for_comp(component.component_name))
    
    # Rotate the image according to its orientation
    rotated_image = component_image.rotate(component.orientation - 90, expand=True)
    
    # Calculate the top-left corner position
    x, y = component.position.x, component.position.y
    top_left_x = x - rotated_image.width // 2
    top_left_y = y - rotated_image.height // 2
    
    # Paste the rotated image onto the final image
    final_image.paste(rotated_image, (top_left_x, top_left_y), rotated_image)
  
  return final_image
  
  
def get_base64_png(circuit_image: Image.Image) -> str:
  # Create a buffer to save the image
  buffered = io.BytesIO()
  # Save the image to the buffer in PNG format
  circuit_image.save(buffered, format="PNG")
  # Encode the buffer content to base64
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_str


def get_image_from_base64(base64_str: str) -> Image.Image:
    # Decode the Base64 string to binary data
    img_data = base64.b64decode(base64_str)
    # Load the binary data into a BytesIO buffer
    buffered = io.BytesIO(img_data)
    # Open the buffer as an image using PIL
    img = Image.open(buffered)
    return img


def merge_images_vertically(img1: Image.Image, img2: Image.Image, margin: int = 10) -> Image.Image:
  # Get the dimensions of both images
  width1, height1 = img1.size
  width2, height2 = img2.size
  
  # Calculate the width and height of the new image, including the margin
  total_width = max(width1, width2)
  total_height = height1 + height2 + margin
  
  # Create a new blank image with black background
  new_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))
  
  # Paste the first image at the top (0, 0)
  new_img.paste(img1, (0, 0))
  
  # Paste the second image below the first image with a margin
  new_img.paste(img2, (0, height1 + margin))
  
  return new_img
