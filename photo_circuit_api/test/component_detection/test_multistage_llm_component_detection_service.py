import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from photocircuit.component_detection.model import CircuitComponents, Component, ComponentPosition, \
  SizedCircuitComponents
from photocircuit.component_detection.multistage_llm_component_detection_service import \
  MultistageLlmComponentDetectionService
from photocircuit.preprocessing.composite_preprocessing_service import CompositePreprocessingService
from photocircuit.preprocessing.scaling_preprocessing_service import ScalingPreprocessingService
from photocircuit.utils.common import base64_to_numpy, numpy_to_base64, scale_image
from photocircuit.utils.component_detection import components_diff
from test.component_detection.utils import dump_array_to_csv
from test.constants import SMALL_CIRCUIT_ID, MEDIUM_CIRCUIT_ID, LARGE_CIRCUIT_ID
from test.report.model import CircuitResult
from test.report.report import generate_report
from test.report.utils import get_generated_circuit, get_base64_png, get_image_from_base64, merge_images_vertically
from test.test_utils import load_circuit_images_with_components, add_labels_to_image, rank_component_detection_err, \
  add_labels_and_bboxs_to_image


class LlmComponentDetectionServiceTest(unittest.TestCase):
  def setUp(self):
    load_dotenv()
    
    self.multistage_llm_component_detection_service = MultistageLlmComponentDetectionService()
    self.preprocessing_service = CompositePreprocessingService(
      ScalingPreprocessingService()
    )
    self.raw_images, self.circuits_components = load_circuit_images_with_components()
    
    # Preprocessing
    # TODO: move to separate file once more tests are made
    self.preprocessed_images, self.preprocessed_circuits_comps = {}, {}
    for circuit_id in self.raw_images.keys():
      raw_circuit_arr = base64_to_numpy(self.raw_images[circuit_id])
      preprocessed_circuit_arr = self.preprocessing_service.preprocess_image(raw_circuit_arr)
      preprocessed_circuit_img = numpy_to_base64(preprocessed_circuit_arr)
      self.preprocessed_images[circuit_id] = preprocessed_circuit_img
      
      raw_circuit_comps = self.circuits_components[circuit_id]
      scale_factor = max(*preprocessed_circuit_arr.shape) / max(*raw_circuit_arr.shape)
      self.preprocessed_circuits_comps[circuit_id] = CircuitComponents(components=[
        Component(
          position=ComponentPosition(
            x=comp.position.x * scale_factor,
            y=comp.position.y * scale_factor
          ),
          component_name=comp.component_name,
          # TODO:
          positive_input_direction=0
        )
        for comp in raw_circuit_comps.components
      ])
    
  def test_one_circuit(self):
    test_circuit_id = LARGE_CIRCUIT_ID
    circuit_comps = self.preprocessed_circuits_comps[test_circuit_id]
    preprocessed_circuit_img = self.preprocessed_images[test_circuit_id]
    
    circuit_img_arr = base64_to_numpy(preprocessed_circuit_img)
    max_len = max(*circuit_img_arr.shape)
    int_size = 60 if max_len <= 600 else 120
    all_circuit_comps_generated = [
      self.multistage_llm_component_detection_service.get_positioned_components(preprocessed_circuit_img, int_size)
      for _ in range(5)
    ]
    combined_comps = []
    for comps_k in all_circuit_comps_generated:
      combined_comps += comps_k.sized_components
    circuit_comps_generated = SizedCircuitComponents(
      sized_components=combined_comps
    )
    labeled_img = add_labels_and_bboxs_to_image(preprocessed_circuit_img, circuit_comps_generated)
    get_image_from_base64(labeled_img).show()
    # avg_error = rank_component_detection_err(
    #   ground_truth_comps=circuit_comps,
    #   predicted_comps=circuit_comps_generated
    # )
    # print(f'avg_error: {avg_error}')
    #
    # generated_circuit_img = get_generated_circuit(circuit_comps_generated)
    #
    # merged_img = merge_images_vertically(generated_circuit_img, get_image_from_base64(preprocessed_circuit_img))
    # merged_img.show()

  def test_detection(self):
    circuit_ids = set(self.circuits_components.keys()).intersection(set(self.raw_images.keys()))
    
    results: list[CircuitResult] = []
    for circuit_id in circuit_ids:
      circuit_comps = self.preprocessed_circuits_comps[circuit_id]
      circuit_img = self.preprocessed_images[circuit_id]
      
      circuit_img_arr = base64_to_numpy(circuit_img)
      max_len = max(*circuit_img_arr.shape)
      int_size = 50 if max_len <= 500 else 100
      circuit_comps_generated = self.llm_component_detection_service.label_components(circuit_img, int_size)
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
          test_image=circuit_img,
          result_image=generated_labeled,
          oriented_img=generated_orientation_img,
          avg_error="{:.2f}".format(avg_error)
        )
      )
    
    generate_report(circuit_results=results)


if __name__ == '__main__':
  unittest.main()
