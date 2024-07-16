import unittest

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from test.report.model import CircuitResult
from test.report.report import generate_report
from test.test_utils import load_circuit_images_with_components, add_labels_to_image


# Define class to test the program
class LlmComponentDetectionServiceTest(unittest.TestCase):
  def setUp(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    self.raw_images, self.circuits_components = load_circuit_images_with_components()
  
  # Function to test addition function
  def test_detection(self):
    # test_circuit_id = "circuit_page_2_circuit_5"
    
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
      
      results.append(
        CircuitResult(
          circuit_id=circuit_id,
          test_image=expected_labeled,
          result_image=generated_labeled,
          overlap="0"
        )
      )
    
    generate_report(circuit_results=results)


if __name__ == '__main__':
  unittest.main()
