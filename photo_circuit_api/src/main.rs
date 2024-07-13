use langchain_rust::llm::{OpenAI, OpenAIModel};

#[macro_use] extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

#[launch]
fn rocket() -> _ {
  // let llm = OpenAI::default().with_model(OpenAIModel::Gpt4o.to_string());

  rocket::build().mount("/", routes![index])
}