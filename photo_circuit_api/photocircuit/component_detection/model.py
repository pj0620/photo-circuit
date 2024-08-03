from dataclasses import dataclass
from enum import Enum

import numpy as np
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
  x: int = Field(description="x location of center of component")
  y: int = Field(description="y location of center of component")
  
  def as_numpy(self):
    return np.array([
      self.x,
      self.y
    ])


class Component(BaseModel):
  position: ComponentPosition = Field(description="center of component in image")
  component_name: ComponentName = Field(description="name of component")
  orientation: int = Field(description="angle from right direction of positive port in degress. For non-polar, choose any port as positive port."
                                       "point to the right.",
                           examples=[0, 90, 270])


class CircuitComponents(BaseModel):
  components: list[Component] = Field(description="list of components in this image")
