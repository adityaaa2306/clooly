"""Silence Signal Detector — identifies prolonged silence as a confusion indicator."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Silence thresholds (in seconds)
SILENCE_THRESHOLD_SOFT = 3.0  # 3 second soft signal
SILENCE_THRESHOLD_HARD = 5.0  # 5 second hard signal
SILENCE_THRESHOLD_CRITICAL = 10.0  # 10 second critical signal


@dataclass
class SilenceEvent:
    """Represents a silence signal."""
    severity: str  # "soft", "hard", "critical"
    duration_seconds: float
    last_utterance_type: Optional[str]  # Type of utterance before silence
    timestamp: float
    action_recommended: str  # "acknowledge", "prompt", "intervene"

    def __repr__(self) -> str:
        return (
            f"SilenceEvent(severity={self.severity}, duration={self.duration_seconds:.1f}s, "
            f"action={self.action_recommended})"
        )


class SilenceDetector:
    """
    Monitor gaps in student responses to detect confusion signals.
    
    Rules:
    - If 3s+ no response after question → soft signal "student thinking"
    - If 5s+ no response → hard signal "student confused"
    - If 10s+ no response → critical signal "student stuck"
    """

    def __init__(self):
        self.last_utterance_time: Optional[float] = None
        self.last_utterance_type: Optional[str] = None
        self.last_message_is_question: bool = False
        self.silence_events: list[SilenceEvent] = []

    def record_utterance(self, utterance_type: str, is_question: bool = False) -> None:
        """Record a student utterance to reset silence timer."""
        self.last_utterance_time = time.time()
        self.last_utterance_type = utterance_type
        self.last_message_is_question = is_question
        logger.debug(f"Recorded utterance: {utterance_type}")

    def record_question_asked(self) -> None:
        """Record that a question was asked (system or student)."""
        self.last_message_is_question = True
        self.last_utterance_time = time.time()
        logger.debug("Question recorded")

    def check_silence(self) -> Optional[SilenceEvent]:
        """
        Check for prolonged silence since last utterance.
        
        Returns a SilenceEvent if threshold exceeded, None otherwise.
        Only returns an event if the last message was a question
        (silence is only meaningful after asking something).
        """
        if self.last_utterance_time is None:
            return None

        if not self.last_message_is_question:
            # Only detect silence after a question
            return None

        current_time = time.time()
        silence_duration = current_time - self.last_utterance_time

        # Check thresholds in order of severity
        if silence_duration >= SILENCE_THRESHOLD_CRITICAL:
            event = SilenceEvent(
                severity="critical",
                duration_seconds=silence_duration,
                last_utterance_type=self.last_utterance_type,
                timestamp=current_time,
                action_recommended="intervene"
            )
            self._log_event(event)
            return event

        elif silence_duration >= SILENCE_THRESHOLD_HARD:
            event = SilenceEvent(
                severity="hard",
                duration_seconds=silence_duration,
                last_utterance_type=self.last_utterance_type,
                timestamp=current_time,
                action_recommended="prompt"
            )
            self._log_event(event)
            return event

        elif silence_duration >= SILENCE_THRESHOLD_SOFT:
            event = SilenceEvent(
                severity="soft",
                duration_seconds=silence_duration,
                last_utterance_type=self.last_utterance_type,
                timestamp=current_time,
                action_recommended="acknowledge"
            )
            self._log_event(event)
            return event

        return None

    def _log_event(self, event: SilenceEvent) -> None:
        """Log and store silence event."""
        self.silence_events.append(event)
        logger.warning(f"Silence detected: {event}")

    def get_silence_summary(self) -> dict:
        """Get summary statistics of silence events in this session."""
        if not self.silence_events:
            return {
                "total_events": 0,
                "average_duration": 0,
                "severity_breakdown": {}
            }

        total_duration = sum(e.duration_seconds for e in self.silence_events)
        avg_duration = total_duration / len(self.silence_events)

        severity_breakdown = {}
        for event in self.silence_events:
            severity_breakdown[event.severity] = severity_breakdown.get(event.severity, 0) + 1

        return {
            "total_events": len(self.silence_events),
            "average_duration": avg_duration,
            "severity_breakdown": severity_breakdown,
            "last_event": self.silence_events[-1] if self.silence_events else None
        }

    def reset(self) -> None:
        """Reset silence detector for new topic/session."""
        self.last_utterance_time = None
        self.last_utterance_type = None
        self.last_message_is_question = False
        self.silence_events = []
        logger.debug("Silence detector reset")
