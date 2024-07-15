def load_prompt(prompt: str) -> str:
  with open('photocircuit/prompts/' + prompt) as f:
    return f.read()
