#[cfg(test)]
mod llm_component_detection_tests {
  use crate::{photo_circuit_api::component_detection::*, test_utils::global_test_setup};

  use llm_component_detection::LlmComponentDetection;
  use component_detection_service::ComponentionDetectionService;

  #[tokio::test]
  async fn test_divide() {
    global_test_setup();
    let llm_component_detection = LlmComponentDetection::default();

    assert_eq!(llm_component_detection.detect_components().await, Ok(String::from("couple of em")));
  }
}

