from photocircuit.component_detection.model.circuit_image import CircuitComponentsLLM, CircuitComponents, BBox


def calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(bbox1.x, bbox2.x)
  yA = max(bbox1.y, bbox2.y)
  xB = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
  yB = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)
  
  # Compute the area of intersection rectangle
  interArea = max(0, xB - xA) * max(0, yB - yA)
  
  # Compute the area of both the prediction and ground-truth rectangles
  box1Area = bbox1.w * bbox1.h
  box2Area = bbox2.w * bbox2.h
  
  # Compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the intersection area
  iou = interArea / float(box1Area + box2Area - interArea)
  
  return iou


def measure_overlap(
    components1: CircuitComponentsLLM | CircuitComponents,
    components2: CircuitComponentsLLM | CircuitComponents) -> float:
  total_iou = 0
  count = 0
  
  for comp1 in components1.components:
    for comp2 in components2.components:
      iou = calculate_iou(comp1.bbox, comp2.bbox)
      total_iou += iou
      count += 1
  
  # Return the average IoU or total overlap
  if count > 0:
    return total_iou / count
  else:
    return 0.0
