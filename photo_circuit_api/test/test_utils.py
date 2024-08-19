import base64
import io
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from langchain.output_parsers import YamlOutputParser

from photocircuit.component_detection.model import Component, CircuitComponents, SizedCircuitComponents
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
            component_name=comp_with_bbox.component_name,
            # TODO:
            positive_input_direction=0,
            id="T"
          )
          for comp_with_bbox in circuit_components.components
        ]
        circuits_with_components[circuit_id] = CircuitComponents(
          components=fixed_comps
        )
        
  common_ids = set(raw_images.keys()).intersection(circuits_with_components.keys())
  raw_images = {k: v for k, v in raw_images.items() if k in common_ids}
  circuits_with_components = {k: v for k, v in circuits_with_components.items() if k in common_ids}
  
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


def get_worst_dist(loc: np.array, image_shape: tuple[int, ...]):
  image_shape_arr = np.array(image_shape)[:len(loc)]
  combined_dists = np.array([loc, image_shape_arr - loc])
  return np.linalg.norm(combined_dists.max(axis=0))


# use to add non-linear penalty for farther distance from true value
def dist_weight(dist):
  return dist


def rank_component_detection_err(ground_truth_comps: CircuitComponents, predicted_comps: CircuitComponents):
  # either there is an extra component/missing component just give it a score of zero
  if len(ground_truth_comps.components) != len(predicted_comps.components):
    return float('inf')

  # map of component name to ground truth positions
  ground_truth_positions_by_name = {}
  for comp in ground_truth_comps.components:
    ground_truth_positions_by_name.setdefault(comp.component_name, [])
    ground_truth_positions_by_name.get(comp.component_name).append(comp.position.as_numpy())

  sum_dists = 0
  for comp in predicted_comps.components:
    loc_arr = comp.position.as_numpy()
    true_locs = ground_truth_positions_by_name.get(comp.component_name)

    # incorrect component, give worst possible dist
    if true_locs is None:
      return float('inf')

    min_dist = min(
      np.linalg.norm(loc_arr - true_loc)
      for true_loc in true_locs
    )
    sum_dists += dist_weight(min_dist)
  return sum_dists / len(ground_truth_comps.components)


def calculate_distance(loc1, loc2):
  return np.linalg.norm(loc1 - loc2)


def rank_component_detection(ground_truth_comps: CircuitComponents, predicted_comps: CircuitComponents,
                             image_shape: tuple[int, ...]):
  total_score = 0
  total_locations = len(ground_truth_comps.components)
  
  # Create dictionaries for easier lookup
  ground_truth_dict = {comp.component_name: comp.position.as_numpy() for comp in ground_truth_comps.components}
  predicted_dict = {comp.component_name: comp.position.as_numpy() for comp in predicted_comps.components}
  
  # Find the maximum possible distance for normalization (e.g., diagonal of the bounding box)
  # min_x = float('inf')
  # min_y = float('inf')
  # max_x = float('-inf')
  # max_y = float('-inf')
  # for l1 in ground_truth_comps.components:
  #   min_x = min(min_x, l1.position.x)
  #   min_y = min(min_y, l1.position.y)
  #   max_x = max(max_x, l1.position.x)
  #   max_y = max(max_y, l1.position.y)
  # max_possible_distance = ((max_y - min_y)**2 + (max_x - min_x)**2)**0.5
  max_possible_distance = (image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5
  
  for component_name, true_loc in ground_truth_dict.items():
    if component_name in predicted_dict:
      guessed_loc = predicted_dict[component_name]
      distance = calculate_distance(guessed_loc, true_loc)
      normalized_distance = distance / max_possible_distance
      score = 1 / (1 + normalized_distance)  # Adding 1 to avoid division by zero
      total_score += score
  
  percent_score = (total_score / total_locations) * 100
  return percent_score
  
  
def add_labels_and_bboxs_to_image(base64_image: str, circuit_components: SizedCircuitComponents) -> str:
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
  
  # Add text labels and bounding boxes to the image
  for component in circuit_components.sized_components:
    position = component.position
    text = component.component_name.value
    approximate_size = component.approximate_size
    
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
    
    # Draw the bounding box
    bbox_top_left = (position.x - approximate_size // 2, position.y - approximate_size // 2)
    bbox_bottom_right = (position.x + approximate_size // 2, position.y + approximate_size // 2)
    draw.rectangle([bbox_top_left, bbox_bottom_right], outline="red", width=2)
  
  # Save the edited image to a BytesIO object
  output_buffer = BytesIO()
  image.save(output_buffer, format="PNG")
  byte_data = output_buffer.getvalue()
  
  # Encode the image back to base64
  base64_encoded_result = base64.b64encode(byte_data).decode('utf-8')
  
  return base64_encoded_result


# def rank_component_detection_avg_err(ground_truth_comps: CircuitComponents, predicted_comps: CircuitComponents,
#                                      image_shape: tuple[int, ...]):
#   total_score = 0
#   total_locations = len(ground_truth_comps.components)
#
#   # Create dictionaries for easier lookup
#   ground_truth_dict = {comp.component_name: comp.position.as_numpy() for comp in ground_truth_comps.components}
#   predicted_dict = {comp.component_name: comp.position.as_numpy() for comp in predicted_comps.components}
#
#   for component_name, true_loc in ground_truth_dict.items():
#     if component_name in predicted_dict:
#       guessed_loc = predicted_dict[component_name]
#       distance = calculate_distance(guessed_loc, true_loc)
#       normalized_distance = distance / max_possible_distance
#       score = 1 / (1 + normalized_distance)  # Adding 1 to avoid division by zero
#       total_score += score
#
#   percent_score = (total_score / total_locations) * 100
#   return percent_score
