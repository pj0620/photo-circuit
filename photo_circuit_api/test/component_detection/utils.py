from photocircuit.component_detection.model import ComponentPosition
from test.component_detection.model import BBox


def bbox_center(bbox: BBox) -> ComponentPosition:
  return ComponentPosition(
    x=bbox.x + bbox.w // 2,
    y=bbox.y + bbox.h // 2
  )
