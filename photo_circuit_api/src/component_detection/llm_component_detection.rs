use langchain_rust::{chain::{Chain, LLMChain, LLMChainBuilder}, fmt_message, fmt_template, llm::{OpenAI, OpenAIModel}, message_formatter, prompt::HumanMessagePromptTemplate, prompt_args, schemas::Message, template_fstring};

use crate::utils::prompt_utils::load_prompt;

use super::component_detection_service::ComponentionDetectionService;
use async_trait::async_trait;

pub struct LlmComponentDetection {
  chain: LLMChain
}

impl Default for LlmComponentDetection {
  fn default() -> Self {
    let llm = OpenAI::default()
      .with_model(OpenAIModel::Gpt4o.to_string());
    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(load_prompt("component_detection/system.txt"))),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        )))
    ];
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm)
        .build()
        .expect("error in chain initialization");
    LlmComponentDetection { chain }
  } 
}

#[async_trait]
impl ComponentionDetectionService for  LlmComponentDetection {
  async fn detect_components(&self, base64_image: &str) -> Result<String, String> {
    match self.chain
      .invoke(prompt_args! {"input" => "Albert Einstein"})
      .await 
    {
        Ok(resp) => Ok(resp),
        Err(e) => Err(e.to_string())
    }
  }
}
