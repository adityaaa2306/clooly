"""Dialogue State Tracker — maintains the conversation context and learning state."""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConfidenceSignals(BaseModel):
    """Track confidence indicators from student responses."""
    correct_count: int = 0
    incorrect_count: int = 0
    confused_count: int = 0
    clarification_requested_count: int = 0

    @property
    def confidence_level(self) -> str:
        """Infer confidence level from signals."""
        total = sum([self.correct_count, self.incorrect_count, self.confused_count])
        if total == 0:
            return "unknown"
        correct_ratio = self.correct_count / total
        if correct_ratio >= 0.8:
            return "confident"
        elif correct_ratio >= 0.5:
            return "mixed"
        else:
            return "confused"


class Misconception(BaseModel):
    """A single misconception identified in student responses."""
    concept: str
    misconception_text: str
    correct_concept: str
    identified_at: float = Field(default_factory=time.time)
    correction_attempts: int = 0


class DialogueState(BaseModel):
    """Complete dialogue state for learning session."""
    
    # Topic context
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    
    # Question tracking
    current_question: Optional[str] = None
    question_history: list[str] = Field(default_factory=list)
    last_question_timestamp: Optional[float] = None
    
    # Explanation history
    last_3_explanations: list[dict] = Field(default_factory=list)  # [{topic, explanation_hash, timestamp}]
    
    # Misconceptions
    identified_misconceptions: list[Misconception] = Field(default_factory=list)
    
    # Confidence tracking
    confidence_signals: ConfidenceSignals = Field(default_factory=ConfidenceSignals)
    
    # Exchange tracking
    exchange_count: int = 0
    last_response_type: Optional[str] = None  # "explanation", "scaffolding", "validation", "correction"
    
    # Session metadata
    session_start_time: float = Field(default_factory=time.time)
    last_update: float = Field(default_factory=time.time)
    
    class Config:
        arbitrary_types_allowed = True

    def add_explanation(self, topic: str, explanation: str) -> None:
        """Track an explanation given to student."""
        explanation_hash = hashlib.md5(explanation.encode()).hexdigest()
        self.last_3_explanations.append({
            "topic": topic,
            "hash": explanation_hash,
            "timestamp": time.time(),
            "text": explanation[:200]  # Store first 200 chars for reference
        })
        # Keep only last 3
        if len(self.last_3_explanations) > 3:
            self.last_3_explanations = self.last_3_explanations[-3:]
        self.last_update = time.time()

    def add_misconception(self, concept: str, misconception_text: str, correct_concept: str) -> None:
        """Track a misconception identified in student response."""
        m = Misconception(
            concept=concept,
            misconception_text=misconception_text,
            correct_concept=correct_concept
        )
        self.identified_misconceptions.append(m)
        self.last_update = time.time()

    def record_correct_response(self) -> None:
        """Record that student responded correctly."""
        self.confidence_signals.correct_count += 1
        self.exchange_count += 1
        self.last_update = time.time()

    def record_incorrect_response(self) -> None:
        """Record that student responded incorrectly."""
        self.confidence_signals.incorrect_count += 1
        self.exchange_count += 1
        self.last_update = time.time()

    def record_confused_response(self) -> None:
        """Record that student expressed confusion."""
        self.confidence_signals.confused_count += 1
        self.exchange_count += 1
        self.last_update = time.time()

    def record_clarification_request(self) -> None:
        """Record that student asked for clarification."""
        self.confidence_signals.clarification_requested_count += 1
        self.exchange_count += 1
        self.last_update = time.time()

    def set_question(self, question: str) -> None:
        """Set current question and track in history."""
        self.current_question = question
        self.question_history.append(question)
        self.last_question_timestamp = time.time()
        self.last_update = time.time()

    def session_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return (time.time() - self.session_start_time) / 60

    def time_since_last_question(self) -> Optional[float]:
        """Get seconds since last question was asked."""
        if self.last_question_timestamp is None:
            return None
        return time.time() - self.last_question_timestamp

    def has_repeated_explanation(self, explanation: str) -> bool:
        """Check if this exact explanation was recently given."""
        explanation_hash = hashlib.md5(explanation.encode()).hexdigest()
        for exp in self.last_3_explanations:
            if exp["hash"] == explanation_hash:
                return True
        return False

    def get_last_misconception_on_topic(self, topic: str) -> Optional[Misconception]:
        """Get most recent misconception identified on this topic."""
        for m in reversed(self.identified_misconceptions):
            if m.concept.lower() == topic.lower():
                return m
        return None

    def reset_for_new_topic(self, new_topic: str, new_subtopic: Optional[str] = None) -> None:
        """Clear state for a new topic while keeping misconceptions."""
        self.topic = new_topic
        self.subtopic = new_subtopic
        self.current_question = None
        self.exchange_count = 0
        self.last_3_explanations = []
        self.confidence_signals = ConfidenceSignals()
        self.last_update = time.time()

    def __repr__(self) -> str:
        return (
            f"DialogueState(topic={self.topic}, subtopic={self.subtopic}, "
            f"exchanges={self.exchange_count}, confidence={self.confidence_signals.confidence_level})"
        )


class DialogueStateManager:
    """Manager wrapper for dialogue state (provides singleton pattern)."""
    
    def __init__(self):
        self.state = DialogueState()
        logger.debug("DialogueStateManager initialized")
    
    def reset(self) -> None:
        """Reset to new dialogue state."""
        self.state = DialogueState()
        logger.debug("DialogueStateManager reset for new session")

