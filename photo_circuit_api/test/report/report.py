import os

from jinja2 import Environment, FileSystemLoader

from test.report.model import CircuitResult


def generate_report(circuit_results: list[CircuitResult]):
  # Initialize Jinja2 environment
  file_path = os.path.abspath(__file__)
  template_dir = os.path.dirname(file_path)
  print('looking for report template in ', template_dir)
  env = Environment(loader=FileSystemLoader(template_dir))
  template = env.get_template('template.html')
  
  # Render the template with the circuit results
  html_content = template.render(circuit_results=circuit_results)
  
  # Save the rendered HTML to a file
  with open('report.html', 'w') as f:
    f.write(html_content)
