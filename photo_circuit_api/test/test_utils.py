import base64
import os
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from langchain.output_parsers import YamlOutputParser

from photocircuit.component_detection.model.circuit_image import CircuitComponents, ComponentLoc, CircuitComponentsLLM
from test.constants import COMP_COLOR_MAP

type json_raw_type = dict[str, json_raw_type | str]


def load_circuit_images_with_components() -> tuple[dict[str, str], list[CircuitComponents]]:
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
  
  parser = YamlOutputParser(pydantic_object=CircuitComponents)
  # load components in images
  circuits_with_components: list[CircuitComponents] = []
  components_path = f"{test_data_dir}/components"
  for filename in os.listdir(components_path):
    if filename.endswith(".yaml"):
      file_path = os.path.join(components_path, filename)
      with open(file_path, "rb") as yaml_file:
        components = parser.parse(yaml_file.read().decode())
        circuits_with_components.append(components)
        
  return raw_images, circuits_with_components

  
def label_image_with_bboxes(base64_image_str: str, circuit_components: CircuitComponentsLLM | CircuitComponents) -> str:
  """
  :param circuit_components: list of components
  :param base64_image_str: circuit image
  :return: image with components labeled
  """
  # Decode the base64 image string
  image_data = base64.b64decode(base64_image_str)
  image = Image.open(BytesIO(image_data))
  
  # Convert image to RGBA if not already in that mode to retain color
  if image.mode != 'RGBA':
    image = image.convert('RGBA')
  
  # Create a draw object
  draw = ImageDraw.Draw(image)
  try:
    font = ImageFont.truetype("arial.ttf", 14)
  except IOError:
    font = ImageFont.load_default()
  
  # Draw each component's bbox and label
  for component in circuit_components.components:
    color = COMP_COLOR_MAP[component.component_name]
    bbox = component.bbox
    component_name = component.component_name.value
    draw.rectangle(
      [(bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h)],
      outline=color,
      width=2
    )
    draw.text((bbox.x, bbox.y - 10), component_name, fill=color, font=font)
  
  # Save the modified image to a bytes buffer
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  
  # Encode the modified image back to a base64 string
  final_base64_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  
  return final_base64_image_str
