"""Repetition Tracker — prevents duplicate explanations, ensures variety."""

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum explanations to track in memory (LRU)
MAX_EXPLANATIONS_IN_MEMORY = 50
# Time window for considering explanations "recent" (in seconds)
RECENT_WINDOW = 600  # 10 minutes


@dataclass
class ExplanationRecord:
    """Record of an explanation given to student."""
    topic: str
    explanation: str
    explanation_hash: str  # MD5 hash
    timestamp: float
    concept_covered: str
    depth_level: str  # "basic", "intermediate", "advanced"

    def is_recent(self) -> bool:
        """Check if this explanation is within the recent window."""
        return (time.time() - self.timestamp) < RECENT_WINDOW

    def age_seconds(self) -> float:
        """Get age of this explanation in seconds."""
        return time.time() - self.timestamp

    def __repr__(self) -> str:
        return f"ExplRecord({self.topic}/{self.concept_covered}, age={self.age_seconds():.0f}s)"


class RepetitionTracker:
    """
    Track explanations given to prevent repetition.
    
    Strategy:
    1. Hash each explanation and store it with metadata
    2. Check if new explanation matches any in recent memory
    3. If match found, request LLM to vary the explanation
    4. Use LRU cache to avoid unbounded memory growth
    """

    def __init__(self):
        self.explanations: deque = deque(maxlen=MAX_EXPLANATIONS_IN_MEMORY)
        self.recent_hashes: dict[str, ExplanationRecord] = {}  # hash -> record
        self.session_start = time.time()

    def add_explanation(
        self,
        topic: str,
        explanation: str,
        concept_covered: str = "",
        depth_level: str = "intermediate"
    ) -> None:
        """
        Record an explanation given to student.
        """
        explanation_hash = self._compute_hash(explanation)
        record = ExplanationRecord(
            topic=topic,
            explanation=explanation,
            explanation_hash=explanation_hash,
            timestamp=time.time(),
            concept_covered=concept_covered,
            depth_level=depth_level
        )
        self.explanations.append(record)
        self.recent_hashes[explanation_hash] = record
        self._cleanup_old_hashes()
        logger.debug(f"Added explanation: {record}")

    def check_repetition(
        self,
        explanation: str,
        topic: str = ""
    ) -> tuple[bool, Optional[ExplanationRecord]]:
        """
        Check if this explanation was recently given.
        
        Returns:
            (is_repeated, original_record)
            is_repeated: True if exact match found in recent memory
            original_record: The original ExplanationRecord if repeated, else None
        """
        explanation_hash = self._compute_hash(explanation)

        if explanation_hash in self.recent_hashes:
            record = self.recent_hashes[explanation_hash]
            if record.is_recent():
                logger.warning(
                    f"Repetition detected: '{explanation[:50]}...' "
                    f"was given {record.age_seconds():.0f}s ago"
                )
                return True, record

        return False, None

    def check_topic_exhaustion(self, topic: str, threshold: int = 5) -> bool:
        """
        Check if we've explained the same topic many times recently.
        
        Returns True if this topic has been explained threshold+ times
        in the recent window.
        """
        recent_topic_count = 0
        for record in self.explanations:
            if record.topic == topic and record.is_recent():
                recent_topic_count += 1

        if recent_topic_count >= threshold:
            logger.info(
                f"Topic exhaustion detected: '{topic}' explained {recent_topic_count} times"
            )
            return True

        return False

    def get_explanation_variants(self, topic: str) -> list[ExplanationRecord]:
        """Get all recent explanations for a topic (to understand what's been covered)."""
        return [r for r in self.explanations if r.topic == topic and r.is_recent()]

    def suggest_depth_increase(self, topic: str) -> bool:
        """
        Suggest going deeper if we've given several basic explanations of the same topic.
        """
        basic_explanations = [
            r for r in self.explanations
            if r.topic == topic and r.depth_level == "basic" and r.is_recent()
        ]
        return len(basic_explanations) >= 3

    def suggest_example_angle(self, topic: str) -> Optional[str]:
        """
        Based on recent explanations, suggest a different angle.
        If we gave a "definition" explanation, next could be "example" or "application".
        """
        variants = self.get_explanation_variants(topic)
        if not variants:
            return None

        # Heuristic: if all recent were conceptual, suggest concrete example
        if all("example" not in r.explanation.lower() for r in variants):
            return "example"

        # If all were examples, suggest conceptual framework
        if all("why" not in r.explanation.lower() for r in variants):
            return "framework"

        return None

    def get_session_summary(self) -> dict:
        """Get summary of explanations in this session."""
        if not self.explanations:
            return {
                "total_explanations": 0,
                "unique_topics": [],
                "session_duration_minutes": (time.time() - self.session_start) / 60
            }

        unique_topics = set(r.topic for r in self.explanations)
        return {
            "total_explanations": len(self.explanations),
            "unique_topics": list(unique_topics),
            "total_unique_topics": len(unique_topics),
            "session_duration_minutes": (time.time() - self.session_start) / 60,
            "explanations_per_topic": {
                topic: len([r for r in self.explanations if r.topic == topic])
                for topic in unique_topics
            }
        }

    def reset(self) -> None:
        """Reset tracker for new session."""
        self.explanations.clear()
        self.recent_hashes.clear()
        self.session_start = time.time()
        logger.debug("Repetition tracker reset")

    def _compute_hash(self, explanation: str) -> str:
        """Compute consistent hash of explanation."""
        # Normalize whitespace to catch near-duplicates
        normalized = " ".join(explanation.split()).lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cleanup_old_hashes(self) -> None:
        """Remove old entries from hash index."""
        to_remove = []
        for exp_hash, record in self.recent_hashes.items():
            if not record.is_recent():
                to_remove.append(exp_hash)
        for exp_hash in to_remove:
            del self.recent_hashes[exp_hash]
