import unittest

from dotenv import load_dotenv
from matplotlib import pyplot as plt

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from test.component_detection.utils import measure_overlap
from test.report.model import CircuitResult
from test.report.report import generate_report
from test.test_utils import load_circuit_images_with_components, label_image_with_bboxes


# Define class to test the program
class LlmComponentDetectionServiceTest(unittest.TestCase):
  def setUp(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    self.raw_images, self.circuits_with_components = load_circuit_images_with_components()
  
  # Function to test addition function
  def test_detection(self):
    test_circuit_id = "circuit_page_2_circuit_5"
    
    circuit_comps = [
      c
      for c in self.circuits_with_components
      if c.circuit_id == test_circuit_id
    ][0]
    circuit_img = self.raw_images[test_circuit_id]
    
    # X = list(range(100, 200, 5))
    # Y = []
    # for int_size in X:
    expected_labeled = label_image_with_bboxes(
      base64_image_str=circuit_img,
      circuit_components=circuit_comps
    )
    
    circuit_comps_generated = self.llm_component_detection_service.label_components(circuit_img, 60)
    generated_labeled = label_image_with_bboxes(
      base64_image_str=circuit_img,
      circuit_components=circuit_comps_generated
    )
    
    overlap = measure_overlap(
      circuit_comps_generated,
      circuit_comps
    )
    # Y.append(overlap)
      
    # print(Y)
    # plt.plot(X, Y)
    # plt.show()
    
    circuit_results = [
      CircuitResult(
        circuit_id=circuit_comps.circuit_id,
        test_image=expected_labeled,
        result_image=generated_labeled,
        overlap=str(round(100 * overlap, 2)) + "%"
      )
    ]
    generate_report(circuit_results=circuit_results)


if __name__ == '__main__':
  unittest.main()
