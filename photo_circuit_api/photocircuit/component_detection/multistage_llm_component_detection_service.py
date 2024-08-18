from langchain.output_parsers import YamlOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from photocircuit.component_detection.llm_component_detection_service import LlmComponentDetectionService
from photocircuit.component_detection.model import CircuitComponents, SizedCircuitComponents
from photocircuit.utils.common import scale_image, base64_to_numpy
from photocircuit.utils.prompt_utils import load_prompt, generate_image_with_grid_base64


class MultistageLlmComponentDetectionService:
  def __init__(self):
    self.llm_component_detection_service = LlmComponentDetectionService()
    self.llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)
    self.parser = YamlOutputParser(pydantic_object=SizedCircuitComponents)
    self.format_instructions = self.parser.get_format_instructions()
    self.system_prompt = load_prompt('multistage_llm_component_detection/system.txt')
    self.chain = self.llm | self.parser
    
  def get_positioned_components(self, base64_circuit_img: str, int_size: int) -> SizedCircuitComponents:
    print('adding gridlines')
    img_with_grid = generate_image_with_grid_base64(base64_circuit_img, int_size, True)
    
    print('invoking gpt4o to label circuit image')
    msgs = [
      SystemMessage(content=self.system_prompt),
      HumanMessage(
        content=[
          {"type": "text", "text": self.format_instructions},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_with_grid}"}},
        ])
    ]
    components = self.chain.invoke(msgs)
    print("got following response from vision llm: \n", components)
    return components
    
  def label_components(self, base64_circuit_img: str, int_size: int) -> CircuitComponents:
    # first_stage_components = self.llm_component_detection_service.label_components(base64_circuit_img, int_size, True)
    # print("[Stage 1] got following response from vision llm: \n", first_stage_components)
    
    print("[Stage 1] Initial Sized Components")
    # first_stage_comps_sized = self.get_positioned_components()
    
    return CircuitComponents(components=[])
