import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from photocircuit.component_detection.model import CircuitComponents, Component, ComponentPosition
from photocircuit.preprocessing.scaling_preprocessing_service import ScalingPreprocessingService
from photocircuit.utils.common import base64_to_numpy, numpy_to_base64, scale_image
from photocircuit.utils.component_detection import components_diff
from test.component_detection.utils import dump_array_to_csv
from test.report.model import CircuitResult
from test.report.report import generate_report
from test.report.utils import get_generated_circuit, get_base64_png, get_image_from_base64, \
  merge_images_vertically
from test.test_utils import load_circuit_images_with_components, add_labels_to_image, rank_component_detection_err


# Define class to test the program
class LlmComponentDetectionServiceTest(unittest.TestCase):
  def setUp(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    self.scaling_preprocessing_service = ScalingPreprocessingService()
    self.raw_images, self.circuits_components = load_circuit_images_with_components()
    
  def test_one_circuit(self):
    test_circuit_id = 'circuit_page_0_circuit_1'
    circuit_comps = self.circuits_components[test_circuit_id]
    raw_circuit_img = self.raw_images[test_circuit_id]
    raw_circuit_arr = base64_to_numpy(raw_circuit_img)
    preprocessed_circuit_arr = self.scaling_preprocessing_service.preprocess_image(raw_circuit_arr)
    preprocessed_circuit_img = numpy_to_base64(preprocessed_circuit_arr)
    circuit_comps_generated = self.llm_component_detection_service.label_components(preprocessed_circuit_img, 50)
    
    avg_error = rank_component_detection_err(
      ground_truth_comps=circuit_comps,
      predicted_comps=circuit_comps_generated
    )
    print(f'avg_error: {avg_error}')
    
    generated_circuit_img = get_generated_circuit(circuit_comps_generated)
    
    merged_img = merge_images_vertically(generated_circuit_img, get_image_from_base64(preprocessed_circuit_img))
    merged_img.show()
  
  # Function to test addition function
  def test_detection(self):
    circuit_ids = set(self.circuits_components.keys()).intersection(set(self.raw_images.keys()))
    
    results: list[CircuitResult] = []
    for circuit_id in circuit_ids:
      circuit_comps = self.circuits_components[circuit_id]
      circuit_img = self.raw_images[circuit_id]
      
      expected_labeled = add_labels_to_image(
        base64_image=circuit_img,
        circuit_components=circuit_comps
      )
  
      circuit_comps_generated = self.llm_component_detection_service.label_components(circuit_img, 60)
      generated_labeled = add_labels_to_image(
        base64_image=circuit_img,
        circuit_components=circuit_comps_generated
      )
      generated_orientation_img = get_base64_png(get_generated_circuit(circuit_comps_generated))

      avg_error = rank_component_detection_err(
        ground_truth_comps=circuit_comps,
        predicted_comps=circuit_comps_generated
      )
      
      results.append(
        CircuitResult(
          circuit_id=circuit_id,
          test_image=expected_labeled,
          result_image=generated_labeled,
          oriented_img=generated_orientation_img,
          avg_error="{:.2f}".format(avg_error)
        )
      )
    
    generate_report(circuit_results=results)
    
  def test_image_size(self):
    test_circuit_id = "circuit_page_2_circuit_3"
    
    circuit_comps = self.circuits_components[test_circuit_id]
    circuit_img = self.raw_images[test_circuit_id]
    
    screen_sizes = np.arange(500, 1000, 100).astype(np.int32)
    intv_sizes = np.arange(5, 105, 5).astype(np.int32)
    
    # screen_sizes = np.arange(500, 600, 100).astype(np.int32)
    # intv_sizes = np.arange(10, 20, 10).astype(np.int32)
    
    # Convert image to a NumPy array
    image_data = base64.b64decode(circuit_img)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    
    max_len = max(image_array.shape)
    res = []
    for i, screen_size in enumerate(screen_sizes):
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
      scaled_circuit_img_arr = scale_image(
        image=image_array,
        screen_size=screen_size
      )
      scaled_circuit_img = numpy_to_base64(scaled_circuit_img_arr)
      for j, intv_size in enumerate(intv_sizes):
        print(scaled_circuit_img)
        circuit_comps_generated = self.llm_component_detection_service.label_components(
          scaled_circuit_img,
          intv_size
        )
        
        avg_error = rank_component_detection_err(
          ground_truth_comps=circuit_comps_scaled,
          predicted_comps=circuit_comps_generated
        )
        
        diff_msg, matches = components_diff(
          expected=circuit_comps_scaled,
          actual=circuit_comps_generated
        )
        if not matches:
          print("# Diff #")
          print(diff_msg)
        
        res.append(
          [test_circuit_id, max_len, screen_sizes[i], intv_sizes[j], avg_error]
        )
  
    print(res)
    header = ["Circuit Id", "Original Size(px)", "Screen Size (px)", "Interval Size (px)", "Average Error (px)"]
    dump_array_to_csv(res, 'screen_sizes_intv_spacing.csv', header=header)


if __name__ == '__main__':
  unittest.main()
