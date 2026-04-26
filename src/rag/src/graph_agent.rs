use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use serde::{Deserialize, Serialize};
use vectradb_storage::PersistentVectorDB;

use crate::llm::{CompletionConfig, LlmProvider};
use crate::RagError;

/// Configuration for the graph-based retrieval agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAgentConfig {
    /// Maximum graph expansion depth.
    pub max_depth: usize,
    /// Maximum branches to explore per node.
    pub branch_factor: usize,
    /// Minimum similarity score to expand a node.
    pub similarity_threshold: f32,
    /// Maximum total documents to explore (safety cap).
    pub max_total_nodes: usize,
    /// Number of initial seed documents.
    pub seed_top_k: usize,
}

impl Default for GraphAgentConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            branch_factor: 3,
            similarity_threshold: 0.3,
            max_total_nodes: 50,
            seed_top_k: 5,
        }
    }
}

/// A node in the retrieval graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalNode {
    pub id: String,
    pub score: f32,
    pub depth: usize,
    pub parent_id: Option<String>,
    pub concepts: Vec<String>,
    pub text_snippet: String,
}

/// Result of a graph-based retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRetrievalResult {
    pub answer: String,
    pub graph: Vec<RetrievalNode>,
    pub reasoning_trace: Vec<String>,
    pub total_documents_explored: usize,
    pub total_time_ms: f64,
}

/// Graph-based retrieval agent that iteratively expands a retrieval graph.
///
/// Algorithm:
/// 1. Embed query → retrieve seed documents
/// 2. For each seed, ask LLM to extract key concepts
/// 3. Embed each concept → search for related documents
/// 4. Repeat expansion up to `max_depth`
/// 5. Synthesize final answer from all retrieved documents
pub struct GraphAgent {
    embedder: Arc<dyn vectradb_embeddings::EmbeddingProvider>,
    db: Arc<RwLock<PersistentVectorDB>>,
    llm: Arc<dyn LlmProvider>,
    config: GraphAgentConfig,
}

impl GraphAgent {
    pub fn new(
        embedder: Arc<dyn vectradb_embeddings::EmbeddingProvider>,
        db: Arc<RwLock<PersistentVectorDB>>,
        llm: Arc<dyn LlmProvider>,
        config: GraphAgentConfig,
    ) -> Self {
        Self {
            embedder,
            db,
            llm,
            config,
        }
    }

    /// Run the graph-based retrieval agent.
    pub async fn query(&self, question: &str) -> Result<GraphRetrievalResult, RagError> {
        let total_start = Instant::now();
        let mut trace = Vec::new();
        let mut graph: Vec<RetrievalNode> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: Vec<(String, Option<String>, usize)> = Vec::new(); // (query_text, parent_id, depth)

        trace.push(format!("Starting graph retrieval for: {question}"));

        // Step 1: Embed and search for seed documents
        let query_vec = self
            .embedder
            .embed(question)
            .await
            .map_err(|e| RagError::EmbeddingError(format!("{e}")))?;

        let query_array = ndarray::Array1::from_vec(query_vec);
        let db = self.db.read().await;
        let seed_results = db
            .search_similar(query_array, self.config.seed_top_k)
            .await
            .map_err(|e| RagError::SearchError(format!("{e}")))?;
        drop(db);

        trace.push(format!("Found {} seed documents", seed_results.len()));

        // Add seeds to graph
        for r in &seed_results {
            if visited.contains(&r.id) {
                continue;
            }
            visited.insert(r.id.clone());

            let text = r
                .metadata
                .tags
                .get("_text")
                .cloned()
                .unwrap_or_else(|| format!("[vector {}]", r.id));

            graph.push(RetrievalNode {
                id: r.id.clone(),
                score: r.score,
                depth: 0,
                parent_id: None,
                concepts: vec![],
                text_snippet: truncate_text(&text, 200),
            });

            // Queue for expansion
            if r.score >= self.config.similarity_threshold {
                frontier.push((text, Some(r.id.clone()), 1));
            }
        }

        // Step 2-4: Iterative expansion
        while !frontier.is_empty() && graph.len() < self.config.max_total_nodes {
            let batch: Vec<_> = std::mem::take(&mut frontier);

            for (doc_text, parent_id, depth) in batch {
                if depth > self.config.max_depth {
                    continue;
                }

                // Extract concepts via LLM
                let concepts = self.extract_concepts(&doc_text, question).await?;
                trace.push(format!(
                    "Depth {depth}: extracted {} concepts from {}",
                    concepts.len(),
                    parent_id.as_deref().unwrap_or("root")
                ));

                // Update parent node with extracted concepts
                if let Some(pid) = &parent_id {
                    if let Some(node) = graph.iter_mut().find(|n| n.id == *pid) {
                        node.concepts = concepts.clone();
                    }
                }

                // Search for each concept
                for concept in concepts.iter().take(self.config.branch_factor) {
                    if graph.len() >= self.config.max_total_nodes {
                        break;
                    }

                    let concept_vec = match self.embedder.embed(concept).await {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let concept_array = ndarray::Array1::from_vec(concept_vec);
                    let db = self.db.read().await;
                    let results = db
                        .search_similar(concept_array, self.config.branch_factor)
                        .await
                        .map_err(|e| RagError::SearchError(format!("{e}")))?;
                    drop(db);

                    for r in &results {
                        if visited.contains(&r.id) || r.score < self.config.similarity_threshold {
                            continue;
                        }
                        visited.insert(r.id.clone());

                        let text = r
                            .metadata
                            .tags
                            .get("_text")
                            .cloned()
                            .unwrap_or_else(|| format!("[vector {}]", r.id));

                        graph.push(RetrievalNode {
                            id: r.id.clone(),
                            score: r.score,
                            depth,
                            parent_id: parent_id.clone(),
                            concepts: vec![],
                            text_snippet: truncate_text(&text, 200),
                        });

                        // Queue deeper expansion if within limits
                        if depth < self.config.max_depth
                            && graph.len() < self.config.max_total_nodes
                        {
                            frontier.push((text, Some(r.id.clone()), depth + 1));
                        }
                    }
                }
            }
        }

        trace.push(format!("Graph complete: {} nodes explored", graph.len()));

        // Step 5: Synthesize answer from all collected documents
        let answer = self.synthesize_answer(question, &graph).await?;

        Ok(GraphRetrievalResult {
            answer,
            total_documents_explored: graph.len(),
            reasoning_trace: trace,
            graph,
            total_time_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn extract_concepts(
        &self,
        document_text: &str,
        question: &str,
    ) -> Result<Vec<String>, RagError> {
        let prompt = format!(
            "Given this document and question, extract 2-3 key concepts or topics that could help find more relevant information. Return ONLY the concepts, one per line, no numbering.\n\nQuestion: {question}\n\nDocument: {}\n\nKey concepts:",
            truncate_text(document_text, 500)
        );

        let config = CompletionConfig {
            max_tokens: 100,
            temperature: 0.3,
            ..Default::default()
        };

        let response = self.llm.complete(&prompt, &config).await?;

        let concepts: Vec<String> = response
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches(|c: char| {
                        c == '-' || c == '•' || c.is_numeric() || c == '.'
                    })
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty() && l.len() > 2)
            .collect();

        Ok(concepts)
    }

    async fn synthesize_answer(
        &self,
        question: &str,
        graph: &[RetrievalNode],
    ) -> Result<String, RagError> {
        // Sort nodes by score and format context
        let mut sorted_nodes: Vec<&RetrievalNode> = graph.iter().collect();
        sorted_nodes.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut context = String::new();
        for (i, node) in sorted_nodes.iter().take(10).enumerate() {
            let depth_info = if node.depth > 0 {
                format!(" (found via expansion, depth={})", node.depth)
            } else {
                " (direct match)".to_string()
            };
            context.push_str(&format!(
                "[{}] (score={:.4}{}) {}\n",
                i + 1,
                node.score,
                depth_info,
                node.text_snippet
            ));
        }

        let prompt = format!(
            "Based on the following retrieved documents (some found directly, some found through graph expansion), answer the question comprehensively.\n\nDocuments:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        );

        let config = CompletionConfig {
            max_tokens: 1024,
            temperature: 0.7,
            system_prompt: Some("You are a helpful assistant. Synthesize information from multiple sources to provide a comprehensive answer.".to_string()),
            ..Default::default()
        };

        self.llm.complete(&prompt, &config).await
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars])
    }
}
