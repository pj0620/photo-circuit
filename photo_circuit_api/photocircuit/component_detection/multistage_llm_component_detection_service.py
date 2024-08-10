from langchain.output_parsers import YamlOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from photocircuit.component_detection.model import CircuitComponents
from photocircuit.utils.common import scale_image, base64_to_numpy
from photocircuit.utils.prompt_utils import load_prompt, generate_image_with_grid_base64


class MultistageLlmComponentDetectionService:
  def __init__(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    
  def label_components(self, base64_circuit_img: str, int_size: int) -> CircuitComponents:
    first_stage_components = self.llm_component_detection_service.label_components(base64_circuit_img, int_size, True)
    print("[Stage 1] got following response from vision llm: \n", first_stage_components)
    
    return first_stage_components
