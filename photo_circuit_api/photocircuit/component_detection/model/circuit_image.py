from dataclasses import dataclass
from enum import Enum

from pydantic.v1 import BaseModel, Field


class BBox(BaseModel):
  h: int = Field(description="height of bbox for component")
  w: int = Field(description="width of bbox for component")
  x: int = Field(description="x location of top left point of bbox for component")
  y: int = Field(description="y location of top left point of bbox for component")


class ComponentName(Enum):
  RESISTOR = 'resistor'
  CAPACITOR = 'capacitor'
  VOLTAGE_SOURCE = 'voltage source'
  CURRENT_SOURCE = 'current source'
  INDUCTOR = 'inductor'
  DEPENDANT_VOLTAGE_SOURCE = 'dependant voltage source'
  DEPENDANT_CURRENT_SOURCE = 'dependant current source'
  UNKNOWN = 'unknown'


class ComponentLoc(BaseModel):
  bbox: BBox = Field(description="bbox defining location of component in image")
  component_name: ComponentName = Field(description="name of component")


class CircuitComponentsLLM(BaseModel):
  components: list[ComponentLoc] = Field(description="list of components in this image")


class CircuitComponents(BaseModel):
  circuit_id: str = Field(description="id of this circuit image")
  components: list[ComponentLoc] = Field(description="list of components in this image")
