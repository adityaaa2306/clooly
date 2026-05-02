"""Statement Assessment — evaluate correctness and identify misconceptions."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CorrectnesLevel(str, Enum):
    """Correctness assessment of a student statement."""
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"


@dataclass
class StatementAssessment:
    """Assessment result for a student statement."""
    statement: str
    correctness: CorrectnesLevel
    claim_extracted: str
    reasoning: str
    misconception_identified: Optional[str] = None
    correct_concept: Optional[str] = None
    confidence: float = 0.7  # 0-1, how confident is this assessment
    response_strategy: str = "acknowledge"  # "reinforce", "redirect", "Socratic", "validate"

    def __repr__(self) -> str:
        return (
            f"Assessment(claim='{self.claim_extracted}', "
            f"correctness={self.correctness.value}, strategy={self.response_strategy})"
        )


class StatementAssessor:
    """
    Assess student statements for correctness and misconceptions.
    This is a hybrid system: pattern-based + heuristics + LLM fallback.
    """

    def __init__(self):
        self.domain_knowledge = self._build_domain_knowledge()

    def _build_domain_knowledge(self) -> dict:
        """Build a simple domain knowledge base for common CS/tech concepts."""
        return {
            "deadlock": {
                "definition": "state where two or more processes are waiting for each other indefinitely",
                "common_misconceptions": {
                    "just starvation": "starvation is processes not getting resources; deadlock is circular wait",
                    "requires deadlock": "no, can have deadlock without starvation",
                    "easy to fix": "actually requires careful design to prevent"
                }
            },
            "recursion": {
                "definition": "function calling itself with base case to stop",
                "common_misconceptions": {
                    "infinite loop": "recursion stops at base case, not infinite",
                    "always inefficient": "recursion can be efficient (memoization, tail recursion)",
                }
            },
            "tcp": {
                "definition": "reliable, ordered, connection-oriented protocol",
                "common_misconceptions": {
                    "faster than udp": "TCP is slower because it ensures reliability",
                    "always better": "UDP better for latency-sensitive apps (gaming, streaming)",
                }
            },
            "three way handshake": {
                "definition": "SYN, SYN-ACK, ACK to establish TCP connection",
                "common_misconceptions": {
                    "four way": "three way for opening, four way for closing",
                    "no security": "doesn't provide encryption, just connection setup",
                }
            },
            "variable": {
                "definition": "named container holding a value in memory",
                "common_misconceptions": {
                    "name is type": "name is independent of type",
                    "value is permanent": "values can change, that's why it's variable",
                }
            }
        }

    def assess(self, statement: str, context: Optional[str] = None, topic: Optional[str] = None) -> StatementAssessment:
        """Assess a student statement for correctness and misconceptions."""
        statement_lower = statement.lower().strip()

        # Try to extract a clear claim
        claim = self._extract_claim(statement)

        # Check against domain knowledge if topic known
        if topic:
            correctness, misconception, correct_concept = self._check_correctness(
                statement_lower, topic, context
            )
        else:
            correctness = CorrectnesLevel.UNCLEAR
            misconception = None
            correct_concept = None

        # Determine response strategy
        strategy = self._choose_strategy(correctness, misconception)

        # Generate reasoning explanation
        reasoning = self._generate_reasoning(correctness, misconception, correct_concept)

        return StatementAssessment(
            statement=statement,
            correctness=correctness,
            claim_extracted=claim,
            reasoning=reasoning,
            misconception_identified=misconception,
            correct_concept=correct_concept,
            confidence=self._estimate_confidence(correctness),
            response_strategy=strategy
        )

    def _extract_claim(self, statement: str) -> str:
        """Extract the core claim from a statement."""
        # Simple heuristic: look for main verb and object
        statement = statement.replace("I think ", "").replace("I believe ", "")
        statement = statement.replace("The answer is ", "").replace("It's ", "")
        return statement.strip()

    def _check_correctness(self, statement_lower: str, topic: str, context: Optional[str]) -> tuple:
        """
        Check correctness of statement against domain knowledge.
        Returns: (correctness_level, misconception, correct_concept)
        """
        topic_lower = topic.lower()

        # Handle common false statements
        false_patterns = {
            "deadlock": {
                "just starvation": ("incorrect", "just starvation", "circular wait of resources"),
                "same as starvation": ("incorrect", "same as starvation", "circular wait vs resource unavailability"),
            },
            "tcp": {
                "faster than udp": ("incorrect", "faster than udp", "TCP slower due to reliability checks"),
                "udp is always better": ("incorrect", "udp is always better", "UDP better only for latency-critical apps"),
            },
            "recursion": {
                "always slower": ("incorrect", "recursion always slower", "recursion efficient with memoization"),
                "infinite loop": ("incorrect", "infinite loop", "stops at base case"),
            }
        }

        # Check false statements
        for topic_key, patterns in false_patterns.items():
            if topic_key in topic_lower:
                for pattern, (correctness, misconception, correct) in patterns.items():
                    if pattern in statement_lower:
                        return (CorrectnesLevel(correctness), misconception, correct)

        # Check for partially correct patterns
        partial_patterns = {
            "deadlock": {
                "two processes": ("partially_correct", None, None),  # true but incomplete
                "waiting": ("partially_correct", None, None),  # true but needs circular context
            }
        }

        for topic_key, patterns in partial_patterns.items():
            if topic_key in topic_lower:
                for pattern, (correctness, misconception, correct) in patterns.items():
                    if pattern in statement_lower:
                        return (CorrectnesLevel(correctness), misconception, correct)

        # If no clear match, mark as unclear (don't guess)
        return (CorrectnesLevel.UNCLEAR, None, None)

    def _choose_strategy(self, correctness: CorrectnesLevel, misconception: Optional[str]) -> str:
        """Choose response strategy based on correctness."""
        if correctness == CorrectnesLevel.CORRECT:
            return "reinforce"
        elif correctness == CorrectnesLevel.PARTIALLY_CORRECT:
            return "redirect"
        elif correctness == CorrectnesLevel.INCORRECT and misconception:
            return "Socratic"
        else:
            return "clarify"

    def _generate_reasoning(
        self, correctness: CorrectnesLevel, misconception: Optional[str], correct_concept: Optional[str]
    ) -> str:
        """Generate explanation for the assessment."""
        if correctness == CorrectnesLevel.CORRECT:
            return "Correct! This understanding is accurate."
        elif correctness == CorrectnesLevel.PARTIALLY_CORRECT:
            return "You're on the right track, but there's more nuance here."
        elif misconception and correct_concept:
            return f"Common misconception: {misconception}. Actually, {correct_concept}."
        else:
            return "Let me help clarify this concept."

    def _estimate_confidence(self, correctness: CorrectnesLevel) -> float:
        """Estimate confidence in the assessment."""
        # High confidence for clear correct/incorrect, lower for unclear
        confidence_map = {
            CorrectnesLevel.CORRECT: 0.9,
            CorrectnesLevel.INCORRECT: 0.85,
            CorrectnesLevel.PARTIALLY_CORRECT: 0.75,
            CorrectnesLevel.UNCLEAR: 0.5,
        }
        return confidence_map.get(correctness, 0.5)


# Singleton instance
_assessor = None


def get_statement_assessor() -> StatementAssessor:
    """Get or create the statement assessor."""
    global _assessor
    if _assessor is None:
        _assessor = StatementAssessor()
    return _assessor
