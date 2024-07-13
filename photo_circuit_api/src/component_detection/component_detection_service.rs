use async_trait::async_trait;

#[async_trait]
pub trait ComponentionDetectionService {
  async fn detect_components(&self) -> Result<String, String>;
}
