from dataclasses import dataclass
from enum import Enum

from pydantic.v1 import BaseModel, Field


class ComponentName(Enum):
  RESISTOR = 'resistor'
  CAPACITOR = 'capacitor'
  VOLTAGE_SOURCE = 'voltage source'
  CURRENT_SOURCE = 'current source'
  INDUCTOR = 'inductor'
  DEPENDANT_VOLTAGE_SOURCE = 'dependant voltage source'
  DEPENDANT_CURRENT_SOURCE = 'dependant current source'
  UNKNOWN = 'unknown'
  

class ComponentPosition(BaseModel):
  x: int = Field(description="x location of component")
  y: int = Field(description="y location of component")
  
  
class Component(BaseModel):
  position: ComponentPosition = Field(description="position of component in image")
  component_name: ComponentName = Field(description="name of component")
  
  
class CircuitComponents(BaseModel):
  components: list[Component] = Field(description="list of components in this image")

