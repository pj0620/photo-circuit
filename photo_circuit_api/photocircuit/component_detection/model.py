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
  position: ComponentPosition = Field(description="exact center of component")
  component_name: ComponentName = Field(description="name of component")
  positive_input_direction: int = Field(
    description="Angle (in degrees) for the positive terminal of the component. For non-polar elements, any wire "
                "can be positive. 0 degrees: right, 90 degrees: top, 180 degrees: left, 270 degrees: bottom.",
    examples=[0, 90, 180, 270]
  )
  positive_input_direction_reasoning: str = Field(
    description="detailed description of choice for positive_input_direction",
    default=None
  )


class SizedComponent(Component):
  approximate_size: int = Field(description="side length of square that is big enough to fit entire component.")


class SizedCircuitComponents(BaseModel):
  sized_components: list[SizedComponent] = Field(description="list of components with size in the image")


class CircuitComponents(BaseModel):
  components: list[Component] = Field(description="list of components in the image")
