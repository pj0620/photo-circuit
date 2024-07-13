use std::{env, fs};

pub fn load_prompt(prompt: &str) -> String {
  match env::current_dir() {
    Ok(path) => println!("The current working directory is: {}", path.display()),
    Err(e) => println!("Error retrieving the current working directory: {}", e),
  }

  let prompt_path = format!("src/prompts/{}", prompt);
  fs::read_to_string(&prompt_path)
    .unwrap_or_else(|_| panic!("Error reading file: {}", prompt_path))
}
