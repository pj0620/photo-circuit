import os

from photocircuit.component_detection.model import ComponentPosition
from test.component_detection.model import BBox
import pandas as pd


def bbox_center(bbox: BBox) -> ComponentPosition:
  return ComponentPosition(
    x=bbox.x + bbox.w // 2,
    y=bbox.y + bbox.h // 2
  )
  
  
def dump_array_to_csv(array, file_path, header=None):
  # Convert the array to a DataFrame
  df = pd.DataFrame(array)
  
  # Check if the file exists
  if not os.path.isfile(file_path):
    # If the file does not exist, add the header if provided
    if header:
      df.to_csv(file_path, index=False, header=header)
    else:
      df.to_csv(file_path, index=False, header=False)
  else:
    # If the file exists, append the data without writing the header
    df.to_csv(file_path, mode='a', index=False, header=False)