"""Comprehensive Utterance Classifier — classify into 8 types."""

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class UtteranceType(str, Enum):
    """8 major utterance types in educational dialogue."""
    DIRECT_QUESTION = "direct_question"  # "What is deadlock?"
    CLARIFICATION_QUESTION = "clarification_question"  # "Wait, what do you mean?"
    CONFIRMATION_QUESTION = "confirmation_question"  # "So it's like X, right?"
    CORRECT_STATEMENT = "correct_statement"  # "The answer is 42"
    INCORRECT_STATEMENT = "incorrect_statement"  # "I think it's 7"
    FRUSTRATED_STATEMENT = "frustrated_statement"  # "I don't get this at all"
    TOPIC_SHIFT = "topic_shift"  # "Can we do something else?"
    FILLER = "filler"  # "Uh, hmm, okay"


class UtteranceClassification:
    """Classification result with type and confidence."""
    
    def __init__(self, utterance: str, utterance_type: UtteranceType, confidence: float, reasoning: str = ""):
        self.utterance = utterance
        self.type = utterance_type
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"UtteranceClassification(type={self.type.value}, confidence={self.confidence:.2f})"


class UtteranceClassifier:
    """
    Classify student utterances into 8 types.
    Uses pattern-based detection with heuristic scoring.
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> dict:
        """Build regex patterns for each utterance type."""
        return {
            UtteranceType.FILLER: [
                r"^\s*(uh|um|hmm|okay|ok|sure|yeah|yep|nope|right|exactly)\s*$",
                r"^\s*(a|an)\s*$",
                r"^\s*\.\.\.\s*$",
            ],
            UtteranceType.CLARIFICATION_QUESTION: [
                r"^(?:wait|hold on|what),?\s+(what|who|where|when|why|how)",
                r"(what do you mean|clarify|rephrase|say that again)",
                r"(didn\'t follow|didn\'t understand)",
            ],
            UtteranceType.CONFIRMATION_QUESTION: [
                r"(right\?|correct\?|is that right\?|is that correct\?)",
                r"so\s+.*\?",
                r"(isn\'?t it|doesn\'?t it|aren\'?t they)",
                r"(am i right|did i get that)",
            ],
            UtteranceType.FRUSTRATED_STATEMENT: [
                r"(don\'?t|can\'?t|never)\s+(get|understand|follow|figure out)",
                r"(confused|stuck|lost|struggling|frustrated)",
                r"(this is hard|too hard|impossible|don\'?t understand|makes no sense)",
                r"(why is this|what\'?s the point)",
            ],
            UtteranceType.TOPIC_SHIFT: [
                r"(can we|let\'?s|could we|should we)\s+(do|try|move on to|switch to|change to)",
                r"(different topic|something else|next topic)",
                r"(skip|move forward|go to)",
            ],
            UtteranceType.DIRECT_QUESTION: [
                r"^(what|who|when|where|why|how|which|is|are|can|could|would|should|do|does|did)\s+",
                r"\?$",
                r"(explain|define|describe|tell me about|give me|show me)\s+",
            ],
        }

    def classify(self, utterance: str, context: Optional[str] = None) -> UtteranceClassification:
        """
        Classify an utterance into one of 8 types.
        Returns UtteranceClassification with type and confidence.
        """
        utterance_lower = utterance.lower().strip()

        # Remove punctuation variants for matching
        utterance_clean = utterance_lower.replace("?", " ").strip()

        # Rule 1: Check filler (highest priority — almost pure filler)
        if self._check_pattern(utterance_clean, UtteranceType.FILLER):
            return UtteranceClassification(
                utterance, UtteranceType.FILLER, confidence=0.95,
                reasoning="Pure filler / backchannel"
            )

        # Rule 2: Check clarification (high specificity)
        if self._check_pattern(utterance_clean, UtteranceType.CLARIFICATION_QUESTION):
            return UtteranceClassification(
                utterance, UtteranceType.CLARIFICATION_QUESTION, confidence=0.90,
                reasoning="Asks for clarification of previous statement"
            )

        # Rule 3: Check confirmation (high specificity)
        if self._check_pattern(utterance_clean, UtteranceType.CONFIRMATION_QUESTION):
            return UtteranceClassification(
                utterance, UtteranceType.CONFIRMATION_QUESTION, confidence=0.88,
                reasoning="Seeks confirmation of understanding"
            )

        # Rule 4: Check topic shift (high specificity)
        if self._check_pattern(utterance_clean, UtteranceType.TOPIC_SHIFT):
            return UtteranceClassification(
                utterance, UtteranceType.TOPIC_SHIFT, confidence=0.85,
                reasoning="Requests topic change"
            )

        # Rule 5: Check frustration (high specificity)
        if self._check_pattern(utterance_clean, UtteranceType.FRUSTRATED_STATEMENT):
            return UtteranceClassification(
                utterance, UtteranceType.FRUSTRATED_STATEMENT, confidence=0.92,
                reasoning="Expresses frustration or confusion"
            )

        # Rule 6: Check direct question (ends in ?)
        if utterance.strip().endswith("?"):
            return UtteranceClassification(
                utterance, UtteranceType.DIRECT_QUESTION, confidence=0.85,
                reasoning="Syntactic question marker (?)"
            )

        # Rule 7: Check for question words at start
        if self._check_pattern(utterance_clean, UtteranceType.DIRECT_QUESTION):
            return UtteranceClassification(
                utterance, UtteranceType.DIRECT_QUESTION, confidence=0.80,
                reasoning="Question word at start (what, how, why, etc.)"
            )

        # Rule 8: Heuristic fallback based on length and negativity
        # Longer utterances with negative sentiment → likely incorrect statement
        if len(utterance.split()) > 4 and any(w in utterance_lower for w in ["think", "believe", "reckon"]):
            # Could be correct or incorrect; use sentiment heuristic
            negative_words = ["not", "no", "don't", "doesn't", "can't", "wrong", "bad"]
            if any(w in utterance_lower for w in negative_words):
                return UtteranceClassification(
                    utterance, UtteranceType.INCORRECT_STATEMENT, confidence=0.60,
                    reasoning="Contains negative sentiment + opinion marker"
                )
            else:
                return UtteranceClassification(
                    utterance, UtteranceType.CORRECT_STATEMENT, confidence=0.65,
                    reasoning="Contains affirmative + opinion marker"
                )

        # Rule 9: Default fallback (neutral statement)
        return UtteranceClassification(
            utterance, UtteranceType.CORRECT_STATEMENT, confidence=0.50,
            reasoning="Neutral statement (no strong signals)"
        )

    def _check_pattern(self, utterance: str, utterance_type: UtteranceType) -> bool:
        """Check if utterance matches any pattern for the type."""
        if utterance_type not in self.patterns:
            return False
        for pattern in self.patterns[utterance_type]:
            if re.search(pattern, utterance):
                return True
        return False


# Singleton instance
_classifier = None


def get_utterance_classifier() -> UtteranceClassifier:
    """Get or create the utterance classifier."""
    global _classifier
    if _classifier is None:
        _classifier = UtteranceClassifier()
    return _classifier
