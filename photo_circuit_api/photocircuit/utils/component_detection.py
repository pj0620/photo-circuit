from typing import Tuple

from photocircuit.component_detection.model import ComponentName, CircuitComponents


def comp_counts(components: CircuitComponents) -> dict[ComponentName, int]:
  comp_counts_map = {}
  for comp in components.components:
    comp_name = comp.component_name
    comp_counts_map.setdefault(comp_name, 0)
    comp_counts_map[comp_name] += 1
  return comp_counts_map


def summary(components: CircuitComponents):
  comp_count_summary = [
    f"{comp_name}: {comp_count}"
    for comp_name, comp_count in comp_counts(components).items()
  ]
  return (
    f"detected {len(components.components)} components\n"
    f"component counts: {"\n".join(comp_count_summary)}"
  )


def components_diff(expected: CircuitComponents, actual: CircuitComponents) -> Tuple[str, bool]:
  """
  Compares two CircuitComponents objects based on the counts of their components.

  Args:
      expected (CircuitComponents): The expected CircuitComponents object.
      actual (CircuitComponents): The actual CircuitComponents object.

  Returns:
      Tuple[bool, Optional[str]]: diff output string, and if outputs match
  """
  expected_counts = comp_counts(expected)
  actual_counts = comp_counts(actual)
  
  if expected_counts != actual_counts:
    diff_output = "Differences found in component counts:\n"
    all_component_names = set(expected_counts.keys()).union(actual_counts.keys())
    for component_name in all_component_names:
      expected_count = expected_counts.get(component_name, 0)
      actual_count = actual_counts.get(component_name, 0)
      if expected_count != actual_count:
        diff_output += f"{component_name.value}: Expected {expected_count}, Found {actual_count}\n"
    return diff_output, False
  
  return "No difference", True
