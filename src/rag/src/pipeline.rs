use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use serde::{Deserialize, Serialize};
use vectradb_components::SimilarityResult;
use vectradb_storage::PersistentVectorDB;

use crate::llm::{ChatMessage, ChatRole, CompletionConfig, LlmProvider};
use crate::RagError;

/// RAG pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Number of documents to retrieve.
    pub top_k: usize,
    /// System prompt for the LLM.
    pub system_prompt: String,
    /// Template for formatting retrieved context. Use `{context}` and `{question}` placeholders.
    pub context_template: String,
    /// Maximum number of context characters to send to the LLM.
    pub max_context_chars: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            system_prompt: "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so.".to_string(),
            context_template: "Context:\n{context}\n\nQuestion: {question}".to_string(),
            max_context_chars: 8000,
        }
    }
}

/// Response from the RAG pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<SourceDocument>,
    pub total_time_ms: f64,
    pub embedding_time_ms: f64,
    pub search_time_ms: f64,
    pub llm_time_ms: f64,
}

/// A source document used in the RAG response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    pub id: String,
    pub score: f32,
    pub metadata: std::collections::HashMap<String, String>,
}

/// RAG pipeline: embed query → search → augment → LLM → response.
pub struct RagPipeline {
    embedder: Arc<dyn vectradb_embeddings::EmbeddingProvider>,
    db: Arc<RwLock<PersistentVectorDB>>,
    llm: Arc<dyn LlmProvider>,
    config: RagConfig,
}

impl RagPipeline {
    pub fn new(
        embedder: Arc<dyn vectradb_embeddings::EmbeddingProvider>,
        db: Arc<RwLock<PersistentVectorDB>>,
        llm: Arc<dyn LlmProvider>,
        config: RagConfig,
    ) -> Self {
        Self {
            embedder,
            db,
            llm,
            config,
        }
    }

    /// Run the RAG pipeline: embed the question, search for relevant documents,
    /// format context, and generate an answer via the LLM.
    pub async fn query(&self, question: &str) -> Result<RagResponse, RagError> {
        let total_start = Instant::now();

        // 1. Embed the question
        let embed_start = Instant::now();
        let query_vec = self
            .embedder
            .embed(question)
            .await
            .map_err(|e| RagError::EmbeddingError(format!("{e}")))?;
        let embedding_time_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

        // 2. Search for relevant documents
        let search_start = Instant::now();
        let query_array = ndarray::Array1::from_vec(query_vec);
        let db = self.db.read().await;
        let results = db
            .search_similar(query_array, self.config.top_k)
            .await
            .map_err(|e| RagError::SearchError(format!("{e}")))?;
        drop(db);
        let search_time_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        // 3. Format context from retrieved documents
        let context = self.format_context(&results);
        let prompt = self
            .config
            .context_template
            .replace("{context}", &context)
            .replace("{question}", question);

        // 4. Generate answer via LLM
        let llm_start = Instant::now();
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt,
        }];
        let completion_config = CompletionConfig {
            system_prompt: Some(self.config.system_prompt.clone()),
            ..Default::default()
        };
        let answer = self.llm.chat(&messages, &completion_config).await?;
        let llm_time_ms = llm_start.elapsed().as_secs_f64() * 1000.0;

        let sources = results
            .iter()
            .map(|r| SourceDocument {
                id: r.id.clone(),
                score: r.score,
                metadata: r.metadata.tags.clone(),
            })
            .collect();

        Ok(RagResponse {
            answer,
            sources,
            total_time_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            embedding_time_ms,
            search_time_ms,
            llm_time_ms,
        })
    }

    fn format_context(&self, results: &[SimilarityResult]) -> String {
        let mut context = String::new();
        for (i, r) in results.iter().enumerate() {
            // Use the "_text" tag if present, otherwise show the vector ID and tags
            let text = r.metadata.tags.get("_text").cloned().unwrap_or_else(|| {
                format!("[Document {} (id={}): {:?}]", i + 1, r.id, r.metadata.tags)
            });

            let entry = format!("[{}] (score={:.4}) {}\n", i + 1, r.score, text);
            if context.len() + entry.len() > self.config.max_context_chars {
                break;
            }
            context.push_str(&entry);
        }
        context
    }
}
