from photocircuit.component_detection.model import ComponentName

COMP_COLOR_MAP = {
  ComponentName.UNKNOWN: 'grey',
  ComponentName.INDUCTOR: 'red',
  ComponentName.RESISTOR: 'blue',
  ComponentName.CAPACITOR: 'green',
  ComponentName.CURRENT_SOURCE: 'orange',
  ComponentName.VOLTAGE_SOURCE: 'purple',
  ComponentName.DEPENDANT_VOLTAGE_SOURCE: 'navy',
  ComponentName.DEPENDANT_CURRENT_SOURCE: 'olive',
}