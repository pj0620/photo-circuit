from dataclasses import dataclass
from enum import Enum

from pydantic.v1 import BaseModel, Field

from photocircuit.component_detection.model import ComponentName


class BBox(BaseModel):
  h: int = Field(description="height of bbox for component")
  w: int = Field(description="width of bbox for component")
  x: int = Field(description="x location of top left point of bbox for component")
  y: int = Field(description="y location of top left point of bbox for component")


class TestDataComponentLoc(BaseModel):
  bbox: BBox = Field(description="bbox defining location of component in image")
  component_name: ComponentName = Field(description="name of component")


class TestDataCircuitComponents(BaseModel):
  circuit_id: str = Field(description="id of this circuit image")
  components: list[TestDataComponentLoc] = Field(description="list of components in this image")
