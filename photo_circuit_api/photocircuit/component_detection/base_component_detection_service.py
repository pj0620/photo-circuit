from abc import ABC, abstractmethod
from typing import Any


class BaseComponentDetectionService(ABC):
  @abstractmethod
  def label_components(self, base64_image: str) -> list[Any]:
    """
    :param base64_image: image of circuit
    :return: list of components
    """