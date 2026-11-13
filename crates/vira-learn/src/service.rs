//! Asynchronous Spaced Repetition service integrating FSRS with hexagonal persistence.

use crate::fsrs_scheduler::{CardRating, CardRecord, FsrsScheduler, PreviewIntervals};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vira_ports::store::StorePort;

/// Errors arising during spaced repetition learning service operations.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearnError {
    #[error("Card not found: {0}")]
    CardNotFound(String),
    #[error("Storage failure: {0}")]
    Storage(String),
    #[error("Scheduler failure: {0}")]
    Scheduler(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

impl From<vira_ports::store::StoreError> for LearnError {
    fn from(err: vira_ports::store::StoreError) -> Self {
        LearnError::Storage(err.to_string())
    }
}

/// Outbound/Domain port for asynchronous spaced repetition management.
#[async_trait]
pub trait SpacedRepetitionPort: Send + Sync {
    /// Creates and stores a new flashcard.
    async fn create_card(
        &self,
        front: &str,
        back: &str,
        cefr_level: Option<String>,
    ) -> Result<CardRecord, LearnError>;

    /// Reviews an existing flashcard with a learner rating.
    async fn review_card(
        &self,
        card_id: &str,
        rating: CardRating,
    ) -> Result<CardRecord, LearnError>;

    /// Reviews an existing flashcard with a learner rating at an explicit timestamp.
    async fn review_card_at(
        &self,
        card_id: &str,
        rating: CardRating,
        now_ms: u64,
    ) -> Result<CardRecord, LearnError>;

    /// Retrieves an existing flashcard by ID.
    async fn get_card(&self, card_id: &str) -> Result<Option<CardRecord>, LearnError>;

    /// Lists cards that are due for review as of the current time.
    async fn get_due_cards(&self, limit: usize) -> Result<Vec<CardRecord>, LearnError>;

    /// Lists cards that are due for review as of an explicit timestamp.
    async fn get_due_cards_at(
        &self,
        now_ms: u64,
        limit: usize,
    ) -> Result<Vec<CardRecord>, LearnError>;

    /// Lists all cards up to a limit.
    async fn list_cards(&self, limit: usize) -> Result<Vec<CardRecord>, LearnError>;

    /// Previews next scheduled intervals for all ratings for an existing card.
    async fn preview_intervals(&self, card_id: &str) -> Result<PreviewIntervals, LearnError>;

    /// Total count of flashcards in storage.
    async fn count_cards(&self) -> Result<usize, LearnError>;

    /// Deletes a flashcard by ID.
    async fn delete_card(&self, card_id: &str) -> Result<(), LearnError>;
}

/// Asynchronous service implementation backed by a hexagonal `StorePort`.
pub struct SpacedRepetitionService {
    scheduler: FsrsScheduler,
    store: Arc<dyn StorePort>,
}

impl SpacedRepetitionService {
    /// Creates a service with default FSRS scheduler.
    pub fn new(store: Arc<dyn StorePort>) -> Self {
        Self {
            scheduler: FsrsScheduler::new(),
            store,
        }
    }

    /// Creates a service with a custom FSRS scheduler instance.
    pub fn with_scheduler(store: Arc<dyn StorePort>, scheduler: FsrsScheduler) -> Self {
        Self { scheduler, store }
    }

    /// Returns reference to internal scheduler.
    pub fn scheduler(&self) -> &FsrsScheduler {
        &self.scheduler
    }

    /// Returns reference to underlying store port.
    pub fn store(&self) -> &Arc<dyn StorePort> {
        &self.store
    }
}

#[async_trait]
impl SpacedRepetitionPort for SpacedRepetitionService {
    async fn create_card(
        &self,
        front: &str,
        back: &str,
        cefr_level: Option<String>,
    ) -> Result<CardRecord, LearnError> {
        let mut card = self.scheduler.new_card(front, back);
        card.cefr_level = cefr_level;

        let flashcard = card.to_flashcard();
        self.store.save_flashcard(&flashcard).await?;
        Ok(card)
    }

    async fn review_card(
        &self,
        card_id: &str,
        rating: CardRating,
    ) -> Result<CardRecord, LearnError> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.review_card_at(card_id, rating, now_ms).await
    }

    async fn review_card_at(
        &self,
        card_id: &str,
        rating: CardRating,
        now_ms: u64,
    ) -> Result<CardRecord, LearnError> {
        let maybe_card = self.store.get_flashcard(card_id).await?;
        let flashcard = maybe_card.ok_or_else(|| LearnError::CardNotFound(card_id.to_string()))?;

        let record = CardRecord::from_flashcard(&flashcard);
        let updated = self.scheduler.review_at(&record, rating, now_ms);

        let updated_flashcard = updated.to_flashcard();
        self.store.save_flashcard(&updated_flashcard).await?;

        Ok(updated)
    }

    async fn get_card(&self, card_id: &str) -> Result<Option<CardRecord>, LearnError> {
        let maybe_card = self.store.get_flashcard(card_id).await?;
        Ok(maybe_card.as_ref().map(CardRecord::from_flashcard))
    }

    async fn get_due_cards(&self, limit: usize) -> Result<Vec<CardRecord>, LearnError> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.get_due_cards_at(now_ms, limit).await
    }

    async fn get_due_cards_at(
        &self,
        now_ms: u64,
        limit: usize,
    ) -> Result<Vec<CardRecord>, LearnError> {
        let due_flashcards = self.store.get_due_flashcards(now_ms, limit).await?;
        Ok(due_flashcards
            .iter()
            .map(CardRecord::from_flashcard)
            .collect())
    }

    async fn list_cards(&self, limit: usize) -> Result<Vec<CardRecord>, LearnError> {
        let cards = self.store.list_flashcards(limit).await?;
        Ok(cards.iter().map(CardRecord::from_flashcard).collect())
    }

    async fn preview_intervals(&self, card_id: &str) -> Result<PreviewIntervals, LearnError> {
        let maybe_card = self.store.get_flashcard(card_id).await?;
        let flashcard = maybe_card.ok_or_else(|| LearnError::CardNotFound(card_id.to_string()))?;
        let record = CardRecord::from_flashcard(&flashcard);
        Ok(self.scheduler.preview_intervals(&record))
    }

    async fn count_cards(&self) -> Result<usize, LearnError> {
        Ok(self.store.count_flashcards().await?)
    }

    async fn delete_card(&self, card_id: &str) -> Result<(), LearnError> {
        Ok(self.store.delete_flashcard(card_id).await?)
    }
}
