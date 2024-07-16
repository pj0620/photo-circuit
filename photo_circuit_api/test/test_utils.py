import base64
import os
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from langchain.output_parsers import YamlOutputParser

from photocircuit.component_detection.model import Component, CircuitComponents
from test.component_detection.model import TestDataCircuitComponents
from test.component_detection.utils import bbox_center
from test.constants import COMP_COLOR_MAP

type json_raw_type = dict[str, json_raw_type | str]


def load_circuit_images_with_components() -> tuple[dict[str, str], dict[str, CircuitComponents]]:
  """
  loads CircuitComponents / circuit images from test data
  :return: (list of dicts of circuit_id to base64 circuit image, list of circuit components)
  """
  file_path = os.path.abspath(__file__)
  test_data_dir = os.path.dirname(file_path) + "/../test/test_data"
  print("looking for test data in " + test_data_dir)

  # load images
  raw_images = {}
  images_path = f"{test_data_dir}/circuits_raw"
  for filename in os.listdir(images_path):
    if filename.endswith(".png"):
      file_path = os.path.join(images_path, filename)
      with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        circuit_id = filename.split('.')[0]
        raw_images[circuit_id] = encoded_string
  
  parser = YamlOutputParser(pydantic_object=TestDataCircuitComponents)
  # load components in images
  circuits_with_components: dict[str, CircuitComponents] = {}
  components_path = f"{test_data_dir}/components"
  for filename in os.listdir(components_path):
    if filename.endswith(".yaml"):
      circuit_id = filename.split(".")[0]
      file_path = os.path.join(components_path, filename)
      with open(file_path, "rb") as yaml_file:
        circuit_components: TestDataCircuitComponents = parser.parse(yaml_file.read().decode())
        fixed_comps = [
          Component(
            position=bbox_center(comp_with_bbox.bbox),
            component_name=comp_with_bbox.component_name
          )
          for comp_with_bbox in circuit_components.components
        ]
        circuits_with_components[circuit_id] = CircuitComponents(
          components=fixed_comps
        )
        
  return raw_images, circuits_with_components
  
  
def get_background_color(component_name):
  return COMP_COLOR_MAP.get(component_name, "white")


def add_labels_to_image(base64_image: str, circuit_components: CircuitComponents) -> str:
  # Decode the base64 image
  image_data = base64.b64decode(base64_image)
  image = Image.open(BytesIO(image_data)).convert("RGB")
  
  # Initialize ImageDraw
  draw = ImageDraw.Draw(image)
  
  # Define a font
  try:
    font = ImageFont.truetype("arial.ttf", 16)
  except IOError:
    font = ImageFont.load_default(12)
  
  # Add text labels to the image
  for component in circuit_components.components:
    position = component.position
    text = component.component_name.value
    
    # Calculate text size and background size using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    background_size = (text_size[0] + 4, text_size[1] + 4)
    
    # Calculate background position
    background_position = (position.x - background_size[0] // 2, position.y - background_size[1] // 2)
    background_rect = [background_position,
                       (background_position[0] + background_size[0], background_position[1] + background_size[1])]
    
    # Draw the background rectangle
    draw.rectangle(background_rect, fill=get_background_color(component.component_name))
    
    # Calculate text position
    text_position = (position.x - text_size[0] // 2, position.y - text_size[1] // 2)
    
    # Draw the text
    draw.text(text_position, text, fill="white", font=font)
  
  # Save the edited image to a BytesIO object
  output_buffer = BytesIO()
  image.save(output_buffer, format="PNG")
  byte_data = output_buffer.getvalue()
  
  # Encode the image back to base64
  base64_encoded_result = base64.b64encode(byte_data).decode('utf-8')
  
  return base64_encoded_result
