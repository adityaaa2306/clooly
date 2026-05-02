"""Coreference Resolver — resolves pronouns and ambiguous references to actual concepts."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Common pronouns that need resolution
PRONOUNS = {
    "it", "that", "this", "they", "them", "their", "he", "she", "him", "her",
    "we", "us", "our", "i", "me", "my", "you", "your", "itself", "themselves"
}

# Demonstratives that need resolution
DEMONSTRATIVES = {"that", "this", "these", "those", "the thing", "the concept", "the idea"}


class CoreferenceResolution:
    """Result of resolving a reference."""

    def __init__(
        self,
        original: str,
        resolved_concept: Optional[str] = None,
        confidence: float = 0.0,
        resolution_type: str = "unknown"  # "pronoun", "demonstrative", "ellipsis", "ambiguous"
    ):
        self.original = original
        self.resolved_concept = resolved_concept
        self.confidence = confidence  # 0-1, how confident is this resolution
        self.resolution_type = resolution_type
        self.is_resolved = resolved_concept is not None

    def __repr__(self) -> str:
        return (
            f"Coref('{self.original}' -> '{self.resolved_concept}', "
            f"conf={self.confidence:.2f}, type={self.resolution_type})"
        )


class CoreferenceResolver:
    """
    Resolve pronouns, demonstratives, and ambiguous references.
    
    Strategy:
    1. Extract recent noun phrases from dialogue state
    2. Match pronouns/demonstratives to nearest plausible noun
    3. Use dialogue context (topic, subtopic, misconceptions) to disambiguate
    4. Return confidence score and multiple candidates if ambiguous
    """

    def __init__(self):
        self.recent_concepts: list[str] = []  # Most recent concepts mentioned
        self.sentence_buffer: list[str] = []  # Recent sentences for context

    def update_context(self, sentence: str, extracted_concepts: list[str]) -> None:
        """Update resolver context with new sentence and concepts."""
        self.sentence_buffer.append(sentence)
        if len(self.sentence_buffer) > 10:
            self.sentence_buffer.pop(0)

        self.recent_concepts.extend(extracted_concepts)
        # Keep only recent unique concepts
        seen = set()
        unique = []
        for concept in reversed(self.recent_concepts):
            if concept not in seen:
                unique.append(concept)
                seen.add(concept)
        self.recent_concepts = unique[:20]  # Keep last 20

    def resolve(self, reference: str, topic: Optional[str] = None) -> CoreferenceResolution:
        """
        Resolve a reference (pronoun, demonstrative, or ambiguous) to actual concept.

        Args:
            reference: The pronoun or demonstrative ("it", "that", "this", etc.)
            topic: Current topic context for disambiguation

        Returns:
            CoreferenceResolution with resolved concept (if found) and confidence
        """
        reference_lower = reference.lower().strip()

        # Check if it's a pronoun or demonstrative
        if reference_lower not in PRONOUNS and reference_lower not in DEMONSTRATIVES:
            # Not a reference, return as-is
            return CoreferenceResolution(reference, reference, 1.0, "not_a_reference")

        # Handle special cases
        if reference_lower in {"i", "me", "my", "we", "us", "our", "you", "your"}:
            # First/second person pronouns don't refer to concepts
            return CoreferenceResolution(reference, None, 0.0, "person_pronoun")

        # If we have recent concepts, try to resolve
        if self.recent_concepts:
            resolved, confidence, resolution_type = self._find_best_referent(
                reference_lower, topic
            )
            return CoreferenceResolution(reference, resolved, confidence, resolution_type)

        # No context available
        return CoreferenceResolution(reference, None, 0.0, "no_context")

    def resolve_in_sentence(self, sentence: str, topic: Optional[str] = None) -> str:
        """
        Resolve all references in a sentence.

        Example:
            Input: "It's useful because that helps with the algorithm."
            Output: "Quadratic formula is useful because quadratic formula helps with sorting."
        """
        words = sentence.split()
        resolved_words = []

        for word in words:
            word_lower = word.lower().rstrip(".,!?;:")
            if word_lower in PRONOUNS or word_lower in DEMONSTRATIVES:
                resolution = self.resolve(word_lower, topic)
                if resolution.is_resolved:
                    # Replace with resolved concept
                    punctuation = word[len(word_lower):]
                    resolved_words.append(resolution.resolved_concept + punctuation)
                else:
                    resolved_words.append(word)
            else:
                resolved_words.append(word)

        return " ".join(resolved_words)

    def _find_best_referent(self, pronoun: str, topic: Optional[str]) -> tuple[str, float, str]:
        """
        Find the best concept to refer to.

        Returns:
            (concept, confidence, resolution_type)
        """
        # Demonstratives + singular usually refer to most recent concept
        if pronoun in {"it", "this", "that"}:
            if self.recent_concepts:
                return self.recent_concepts[0], 0.8, "demonstrative_recent"

        # Plural pronouns refer to multiple recent concepts
        if pronoun in {"they", "them", "these", "those"}:
            if len(self.recent_concepts) >= 2:
                return f"{self.recent_concepts[0]} and {self.recent_concepts[1]}", 0.7, "plural_recent"

        # Topic-specific resolution
        if topic and self.recent_concepts:
            # Prefer concepts related to current topic
            for concept in self.recent_concepts:
                if topic.lower() in concept.lower() or concept.lower() in topic.lower():
                    return concept, 0.85, "topic_match"

        # Fallback: most recent concept
        if self.recent_concepts:
            return self.recent_concepts[0], 0.6, "most_recent"

        return None, 0.0, "unresolved"

    def extract_noun_phrases(self, sentence: str) -> list[str]:
        """
        Extract potential noun phrases from a sentence.
        Simple heuristic-based extraction.

        This is a simplified version; in production, use spaCy or similar.
        """
        # Remove common stop words and extract capital-letter phrases or quoted concepts
        phrases = []

        # Pattern 1: Quoted concepts ("concept in quotes")
        quoted = re.findall(r'"([^"]+)"', sentence)
        phrases.extend(quoted)

        # Pattern 2: Capitalized phrases (likely named concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        phrases.extend(capitalized)

        # Pattern 3: Concepts before "is" or "are"
        is_patterns = re.findall(r'(\b[a-z]+(?:\s+[a-z]+)*?)\s+(?:is|are)\s+', sentence, re.IGNORECASE)
        phrases.extend(is_patterns)

        return list(set(phrases))  # Remove duplicates

    def reset(self) -> None:
        """Reset resolver for new context."""
        self.recent_concepts = []
        self.sentence_buffer = []
        logger.debug("Coreference resolver reset")

    def get_context_summary(self) -> dict:
        """Get summary of resolver context."""
        return {
            "recent_concepts": self.recent_concepts[:5],  # Top 5
            "total_concepts_tracked": len(self.recent_concepts),
            "recent_sentences": len(self.sentence_buffer),
        }
