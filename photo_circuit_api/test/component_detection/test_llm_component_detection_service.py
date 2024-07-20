import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from photocircuit.component_detection.model import CircuitComponents, Component, ComponentPosition
from test.report.model import CircuitResult
from test.report.report import generate_report
from test.test_utils import load_circuit_images_with_components, add_labels_to_image, rank_component_detection, \
  rank_component_detection_err


# Define class to test the program
class LlmComponentDetectionServiceTest(unittest.TestCase):
  def setUp(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    self.raw_images, self.circuits_components = load_circuit_images_with_components()
  
  # Function to test addition function
  def test_detection(self):
    circuit_ids = set(self.circuits_components.keys()).intersection(set(self.raw_images.keys()))
    
    results: list[CircuitResult] = []
    for circuit_id in circuit_ids:
      circuit_comps = self.circuits_components[circuit_id]
      circuit_img = self.raw_images[circuit_id]
      
      image_data = base64.b64decode(circuit_img)
      image = Image.open(BytesIO(image_data))
      image_array = np.array(image)
      
      expected_labeled = add_labels_to_image(
        base64_image=circuit_img,
        circuit_components=circuit_comps
      )
  
      circuit_comps_generated = self.llm_component_detection_service.label_components(circuit_img, 60)
      generated_labeled = add_labels_to_image(
        base64_image=circuit_img,
        circuit_components=circuit_comps_generated
      )
      
      avg_error = rank_component_detection_err(
        ground_truth_comps=circuit_comps,
        predicted_comps=circuit_comps_generated,
        image_shape=image_array.shape
      )
      
      results.append(
        CircuitResult(
          circuit_id=circuit_id,
          test_image=expected_labeled,
          result_image=generated_labeled,
          avg_error="{:.2f}".format(avg_error)
        )
      )
    
    generate_report(circuit_results=results)
    
  def test_image_size(self):
    test_circuit_id = "circuit_page_2_circuit_4"
    
    circuit_comps = self.circuits_components[test_circuit_id]
    circuit_img = self.raw_images[test_circuit_id]
    
    screen_sizes = np.arange(100, 1000, 100).astype(np.int32)
    
    # Convert image to a NumPy array
    image_data = base64.b64decode(circuit_img)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    
    max_len = max(image_array.shape)
    for screen_size in screen_sizes:
      scale_factor = screen_size / max_len
      circuit_comps_scaled = CircuitComponents(components=[
        Component(
          position=ComponentPosition(
            x=comp.position.x * scale_factor,
            y=comp.position.y * scale_factor
          ),
          component_name=comp.component_name
        )
        for comp in circuit_comps.components
      ])
    
    
    
    


if __name__ == '__main__':
  unittest.main()
