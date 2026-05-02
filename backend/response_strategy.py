"""Response Strategy Engine — routes responses based on dialogue state and pedagogy."""

import logging
from typing import Optional

from backend.dialogue_state import DialogueState
from backend.statement_assessment import CorrectnesLevel, StatementAssessment
from backend.utterance_classifier import UtteranceType

logger = logging.getLogger(__name__)


class ResponseStrategy:
    """Represents a pedagogical response strategy."""
    
    def __init__(
        self,
        strategy_type: str,
        description: str,
        system_prompt_instruction: str,
        tone: str = "neutral"
    ):
        self.type = strategy_type  # "reinforce", "redirect", "socratic", "explain", "acknowledge", etc.
        self.description = description
        self.instruction = system_prompt_instruction
        self.tone = tone  # "encouraging", "challenging", "supportive", "neutral"

    def __repr__(self) -> str:
        return f"Strategy({self.type}: {self.description})"


class PedagogicalStrategyEngine:
    """
    Route responses based on dialogue state, utterance type, statement assessment,
    and learning science principles.
    """

    def __init__(self):
        self.strategies = self._build_strategies()

    def _build_strategies(self) -> dict:
        """Define pedagogical strategies."""
        return {
            "reinforce_correct": ResponseStrategy(
                "reinforce",
                "Student answered correctly — reinforce and move forward",
                "The student's answer is correct. Affirm their understanding, "
                "then either move to next concept or go deeper if they seem ready.",
                tone="encouraging"
            ),
            "acknowledge_partial": ResponseStrategy(
                "redirect",
                "Student partially correct — acknowledge good parts, gently redirect",
                "The student's answer is partially correct. Acknowledge what they got right, "
                "then clarify the missing or incorrect part. Be encouraging, not critical.",
                tone="supportive"
            ),
            "socratic_incorrect": ResponseStrategy(
                "socratic",
                "Student incorrect — use Socratic questions to guide self-correction",
                "Don't say the answer is wrong. Instead, ask a leading question that helps "
                "them think through their mistake. For example, ask them to test their logic, "
                "or ask 'what happens if...?'",
                tone="neutral"
            ),
            "scaffold_response": ResponseStrategy(
                "explain",
                "Student confused — scaffold with clear, simple steps",
                "The student is confused. Slow down and break the concept into smaller, "
                "simpler pieces. Give a clear, step-by-step framework they can follow.",
                tone="supportive"
            ),
            "clarify_ambiguous": ResponseStrategy(
                "clarify",
                "Ambiguous statement — ask for clarification before assessing",
                "The student's statement is unclear. Ask them to clarify what they mean "
                "before proceeding. Be curious, not judgmental.",
                tone="neutral"
            ),
            "acknowledge_frustration": ResponseStrategy(
                "acknowledge",
                "Student frustrated — validate, simplify, offer encouragement",
                "The student is frustrated. Acknowledge that this concept is tricky, "
                "validate their effort, then offer a simpler breakdown or different angle.",
                tone="encouraging"
            ),
            "handle_topic_shift": ResponseStrategy(
                "acknowledge",
                "Student wants topic shift — acknowledge intent, provide path",
                "The student wants to move on or try something different. Acknowledge that request, "
                "then either honor it or explain why staying on topic might be better.",
                tone="supportive"
            ),
            "avoid_repetition": ResponseStrategy(
                "explain",
                "Student asking about same topic — provide NEW angle, not repeat",
                "The student is asking about something you just explained. Explain it differently "
                "this time — use different examples, different structure, different analogy.",
                tone="neutral"
            ),
            "adjust_difficulty_up": ResponseStrategy(
                "challenge",
                "Student confident — increase challenge / go deeper",
                "The student has demonstrated solid understanding. Challenge them with a harder "
                "variant, or ask them to explain it to someone else, or connect to related concepts.",
                tone="neutral"
            ),
            "adjust_difficulty_down": ResponseStrategy(
                "simplify",
                "Student confused after multiple exchanges — drastically simplify",
                "After multiple exchanges, the student is still confused. Stop and start much simpler. "
                "Use concrete examples, analogies, or hands-on breakdowns.",
                tone="supportive"
            ),
        }

    def choose_strategy(
        self,
        utterance_type: UtteranceType,
        assessment: Optional[StatementAssessment] = None,
        dialogue_state: Optional[DialogueState] = None
    ) -> ResponseStrategy:
        """
        Choose response strategy based on utterance type, assessment, and dialogue state.
        Returns the appropriate ResponseStrategy.
        """
        
        # Route by utterance type first
        if utterance_type == UtteranceType.FILLER:
            # Minimal response; could acknowledge or ignore
            return self.strategies["acknowledge_frustration"]

        if utterance_type == UtteranceType.CLARIFICATION_QUESTION:
            # Student didn't understand previous explanation
            return self.strategies["scaffold_response"]

        if utterance_type == UtteranceType.CONFIRMATION_QUESTION:
            # Student is checking if they understood
            if assessment and assessment.correctness == CorrectnesLevel.CORRECT:
                return self.strategies["reinforce_correct"]
            else:
                return self.strategies["acknowledge_partial"]

        if utterance_type == UtteranceType.FRUSTRATED_STATEMENT:
            # Student is frustrated
            return self.strategies["acknowledge_frustration"]

        if utterance_type == UtteranceType.TOPIC_SHIFT:
            # Student wants to change topic
            return self.strategies["handle_topic_shift"]

        if utterance_type == UtteranceType.DIRECT_QUESTION:
            # Student is asking a new question; answer it
            return self.strategies["scaffold_response"]

        # For statements (CORRECT/INCORRECT/UNCLEAR)
        if utterance_type in [
            UtteranceType.CORRECT_STATEMENT,
            UtteranceType.INCORRECT_STATEMENT
        ]:
            if not assessment:
                # Can't assess without info; clarify
                return self.strategies["clarify_ambiguous"]

            # Route by correctness
            if assessment.correctness == CorrectnesLevel.CORRECT:
                return self._choose_strategy_for_correct(dialogue_state)
            elif assessment.correctness == CorrectnesLevel.PARTIALLY_CORRECT:
                return self.strategies["acknowledge_partial"]
            elif assessment.correctness == CorrectnesLevel.INCORRECT:
                return self.strategies["socratic_incorrect"]
            else:
                return self.strategies["clarify_ambiguous"]

        # Fallback
        return self.strategies["explain"]

    def _choose_strategy_for_correct(self, dialogue_state: Optional[DialogueState]) -> ResponseStrategy:
        """Choose strategy when student answered correctly."""
        if not dialogue_state:
            return self.strategies["reinforce_correct"]

        # If student has been correct multiple times and confident, challenge them
        if (dialogue_state.confidence_signals.correct_count >= 3 and
            dialogue_state.confidence_signals.confidence_level == "confident"):
            return self.strategies["adjust_difficulty_up"]

        # If low confusion count, proceed normally
        if dialogue_state.confidence_signals.confused_count == 0:
            return self.strategies["reinforce_correct"]

        # Mixed signals; reinforce but don't go too deep
        return self.strategies["reinforce_correct"]

    def generate_system_prompt_instruction(
        self,
        dialogue_state: DialogueState,
        strategy: ResponseStrategy
    ) -> str:
        """
        Generate a system prompt instruction combining strategy + context.
        """
        instruction = f"# Pedagogical Strategy: {strategy.description}\n\n"
        instruction += f"{strategy.instruction}\n\n"

        # Add context from dialogue state
        instruction += "# Context:\n"
        instruction += f"- Topic: {dialogue_state.topic or 'unknown'}\n"
        instruction += f"- Student confidence level: {dialogue_state.confidence_signals.confidence_level}\n"
        instruction += f"- Exchanges in this topic: {dialogue_state.exchange_count}\n"

        # Add specific guidance based on state
        if dialogue_state.identified_misconceptions:
            instruction += f"- ATTENTION: Student has {len(dialogue_state.identified_misconceptions)} known misconception(s).\n"
            latest_misconception = dialogue_state.identified_misconceptions[-1]
            instruction += f"  Most recent: They believe '{latest_misconception.misconception_text}' "
            instruction += f"but actually '{latest_misconception.correct_concept}'\n"

        if dialogue_state.exchange_count > 3:
            instruction += f"- PACING: Already {dialogue_state.exchange_count} exchanges. "
            instruction += "Consider whether to move on or drill deeper.\n"

        if dialogue_state.has_repeated_explanation(dialogue_state.current_question or ""):
            instruction += "- REPETITION ALERT: You've covered this before. Explain differently this time.\n"

        instruction += "\n# Response Tone: " + strategy.tone.upper()

        return instruction


# Singleton instance
_engine = None


def get_strategy_engine() -> PedagogicalStrategyEngine:
    """Get or create the strategy engine."""
    global _engine
    if _engine is None:
        _engine = PedagogicalStrategyEngine()
    return _engine
