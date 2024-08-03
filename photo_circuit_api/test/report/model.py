from dataclasses import dataclass


@dataclass
class CircuitResult:
  circuit_id: str
  test_image: str
  result_image: str
  oriented_img: str
  avg_error: str
