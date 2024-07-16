from langchain.output_parsers import YamlOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from photocircuit.component_detection.model import CircuitComponents
from photocircuit.utils.prompt_utils import load_prompt, generate_image_with_grid_base64


# class LlmComponentDetectionService(BaseComponentDetectionService):
class LlmComponentDetectionService:
  def __init__(self):
    self.llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)
    self.parser = YamlOutputParser(pydantic_object=CircuitComponents)
    self.format_instructions = self.parser.get_format_instructions()
    self.system_prompt = load_prompt('component_detection/system.txt')
    self.chain = self.llm | self.parser
    
  def label_components(self, base64_image: str, int_size: int) -> CircuitComponents:
    print('adding gridlines')
    img_with_grid = generate_image_with_grid_base64(base64_image, int_size)
    
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
