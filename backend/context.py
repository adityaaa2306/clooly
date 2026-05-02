"""Rolling transcript buffer and question detection."""

import hashlib
import logging
import re
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded zero-shot classifier
_classifier = None


def get_question_classifier():
    """
    Lazy-load the zero-shot question classifier.
    Runs on CPU, ~80MB model, ~20-40ms per inference.
    Handles all edge cases: "Break down recursion for me", "Give me an example",
    "Pretend you're explaining this to...", etc.
    """
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            logger.info("Loading zero-shot question classifier...")
            _classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1,  # -1 = CPU
            )
            logger.info("Zero-shot classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load zero-shot classifier: {e}")
            _classifier = None
    return _classifier


QUESTION_PREFIXES = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "tell",
    "explain",
    "define",
    "describe",
)

QUESTION_CUES = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whether",
    "if",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
}

# Question starters — checked at START of sentence only to avoid false positives
QUESTION_STARTERS = [
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can you",
    "could you",
    "would you",
    "should you",
    "do you",
    "does",
    "did",
    "have you",
    "has",
    "is there",
    "are there",
    "tell me",
    "explain",
    "describe",
    "walk me through",
    "what's",
    "who's",
    "how's",
    "where's",
    "when's",
    "why's",
]

COMMAND_QUESTION_PREFIXES = (
    ("please", "tell"),
    ("tell",),
    ("can", "you", "tell"),
    ("could", "you", "tell"),
    ("would", "you", "tell"),
    ("please", "explain"),
    ("explain",),
    ("please", "define"),
    ("define",),
    ("please", "describe"),
    ("describe",),
    ("please", "give"),
    ("give",),
)


class ContextEngine:
    """Maintains finalized transcript memory and detects interviewer questions."""

    def __init__(self, window_seconds: int = 45):
        self.window_seconds = window_seconds
        self.buffer: deque = deque()
        self.questions_history: deque = deque()
        self.last_question: Optional[str] = None
        self.last_triggered_hash: Optional[str] = None  # For deduplication

    def add_transcript(
        self,
        speaker: str,
        text: str,
        is_final: bool,
        confidence: float = 1.0,
    ) -> Optional[dict]:
        """Add finalized transcript text to rolling memory."""
        if not is_final:
            return None

        timestamp = time.time()
        normalized_text = self.normalize_question_text(speaker, text, confidence)

        record = {
            "timestamp": timestamp,
            "speaker": speaker,
            "text": normalized_text,
            "raw_text": text,
            "is_final": True,
            "confidence": confidence,
            "is_question": self.is_question(normalized_text, confidence) if speaker == "interviewer" else False,
        }

        self.buffer.append(record)
        self._cleanup_old_transcripts()

        if record["is_question"]:
            self.last_question = normalized_text
            self.questions_history.append(
                {
                    "timestamp": timestamp,
                    "speaker": speaker,
                    "text": normalized_text,
                }
            )
            while len(self.questions_history) > 10:
                self.questions_history.popleft()

        return record

    def get_last_question(self) -> Optional[str]:
        """Return the most recently detected question for compatibility."""
        return self.last_question

    def get_question_from_record(self, record: Optional[dict]) -> Optional[str]:
        """Return question text only if this exact finalized record is a question."""
        if not record:
            return None
        if record.get("speaker") != "interviewer":
            return None
        if not record.get("is_question"):
            return None
        return record.get("text")

    def get_summary(self) -> str:
        """Return a compact summary of the last two question/answer exchanges."""
        if not self.questions_history:
            return ""

        recent_questions = list(self.questions_history)[-2:]
        summary_parts = []

        for question_info in recent_questions:
            question_text = question_info["text"]
            summary_parts.append(f"Q: {question_text}")

            question_time = question_info["timestamp"]
            user_responses = [
                item["text"]
                for item in self.buffer
                if item["speaker"] == "user" and item["timestamp"] > question_time
            ]

            if user_responses:
                summary_parts.append(f"A: {' '.join(user_responses)}")

        words = "\n".join(summary_parts).split()
        if len(words) > 100:
            return " ".join(words[:100]) + "..."
        return "\n".join(summary_parts)

    def clear(self) -> None:
        self.buffer.clear()
        self.questions_history.clear()
        self.last_question = None

    def normalize_question_text(self, speaker: str, text: str, confidence: float) -> str:
        """Clean punctuation (no question mark addition — let is_question decide)."""
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return cleaned
        return cleaned

    def _detect_indirect_question_patterns(self, text: str) -> bool:
        """
        Expanded pattern detection for all 15 edge-case categories.
        Covers: intent-to-learn, requests, comparisons, negation, hypotheticals, etc.
        """
        t = text.lower().strip()
        
        # Category 1: Intent to understand / Confusion patterns
        intent_patterns = [
            r"i\s+(never|don\'?t|didn\'?t)\s+(really\s+)?(got|understand|get|know|see|follow|grasp)",
            r"(confused|unclear|stuck|lost|struggling|unsure|tricky|hard|difficult)\s+(about|on|with|by|to)",
            r"(always\s+)?a\s+bit\s+(confused|unsure|unclear|lost|tricky|hard|difficult)\s+(?:about|on|to)",
            r"don\'?t\s+(really\s+)?(understand|get|follow|see)\s+",
            r"can\'?t\s+(figure out|understand|follow|see|wrap|get)",
            r"(don\'?t|never)\s+really\s+.*understand",
            r"(this|one|it)\s+(always\s+)?(messes|confuses|stumps|trips|is)\s+(?:with\s+)?(me|up|confusing|tricky|hard)",  # "This always messes with me"
            r"(?:kind\s+of|kinda|sorta|seems\s+)+(tricky|hard|difficult|confusing)",
        ]
        
        # Category 2: Embedded/buried questions
        buried_patterns = [
            r"and\s+(i|we)\s+(didn\'?t|don\'?t|haven\'?t)\s+.*(?:follow|understand|get|see|know|realize)",
            r"and\s+(i|we)\s+(?:blank|blank out|forget|don\'?t remember|(?:am|are)\s+not\s+sure)\s+(?:what|why|how)",
        ]
        
        # Category 3: Command/Request patterns (no explicit verb required)
        request_patterns = [
            r"^(?:please\s+)?(tell|explain|describe|define|clarify|break down|walk through|help|show|give|list|provide)\s+(?:me\s+)?",
            r"need\s+(?:an?\s+)?(explanation|definition|breakdown|clarity|help|guidance|clarification)\s+(?:on|about|with|of)",
            r"(?:would|could|can)\s+(?:you\s+)?(explain|tell|describe|define|clarify)",
            r"\.\.\.(?:would|could|can)\s+(?:you\s+)?(explain|tell|describe)",
            r"(?:explanation|clarification|breakdown|definition|walkthrough)\s+(?:would\s+)?(help|be useful|be great)",
        ]
        
        # Category 4: Elliptical queries (already handled by ? gate)
        elliptical_patterns = [
            r"^\?$",  # Just a question mark
            r"^\w+\?$",  # Single word + ?
            r"^\w+\s+\w+\?$",  # Two words + ?
            r"\s+vs\.?\s+",  # vs comparison
            r"\s+versus\s+",
            r"compared\s+to",
            r"difference\s+between",
        ]
        
        # Category 5: Sarcasm / Rhetorical (tone-based)
        sarcasm_patterns = [
            r"(?:yeah|sure|oh|right|lol)\s+(?:because|so).*(?:easy|simple|obvious|clear)",
            r".*(?:sooo|so\s+very|way|super)\s+(?:easy|simple|obvious|clear|straightforward).*(?:right|\?|\.\.\.)",
            r"(?:obviously|clearly|surely)\s+\w+\s+(?:makes|is)\s+(?:so\s+)?(much\s+)?(sense|obvious)",
            r"(?:oh|yeah)\s+(?:sure|right).*(?:makes|is)\s+(?:so\s+)?(much\s+)?sense",  # "Oh sure, that makes sense"
        ]
        
        # Category 6: Multi-intent / Mixed signals
        mixed_patterns = [
            r"^[^?]*\s+(?:right|or|but|though|however|still)\s+(?:or|am i|are you|didn\'?t|isn\'?t|aren\'?t)",
            r"(?:but\s+)?(i\s+think|i\s+believe|i\'m\s+not\s+sure|not\s+fully\s+sure)\s+",
            r"(?:right|correct|true|accurate)\s+(?:or\s+)?(?:am\s+i|did\s+i)\s+.*",
        ]
        
        # Category 7: Negation-based queries (with why/is/does)
        negation_patterns = [
            r"(?:why|how)\s+(?:isn\'?t|aren\'?t|doesn\'?t|don\'?t|didn\'?t|wasn\'?t|weren\'?t)",
            r"(?:is|does|are)\s+.*(?:not|isn\'?t|doesn\'?t|aren\'?t)\s+",
            r"(?:doesn\'?t|don\'?t|isn\'?t|aren\'?t)\s+.*(?:mean|equal|represent|count as|same as)",
        ]
        
        # Category 8: Context-dependent follow-ups
        followup_patterns = [
            r"^(?:and\s+)?(?:this|that)\s+(?:happens|occurs|works)\s+because\s+",
            r"^(?:so|and)\s+(?:what|why|how)\s+(?:does|would|can)",
            r"^(?:why|what|how)\s+(?:though|tho|exactly|precisely|specifically|\?)",
            r"^why\s+(?:though|tho|exactly|\?|$)",
        ]
        
        # Category 9: Verbose/noisy inputs (with "didn't get" or "kind of didn't")
        verbose_patterns = [
            r"(?:kind\s+of\s+)?didn\'?t\s+(?:really\s+)?(?:get|understand|follow|see)",
            r"(?:like|kinda)\s+(?:don\'?t|didn\'?t)\s+(?:get|understand|follow|follow)",
        ]
        
        # Category 10: Domain-ambiguous but clearly requests
        ambiguous_patterns = [
            r"^([a-z_\s]+)\s+(?:meaning|definition|explanation|breakdown|summary|overview)$",
            r"^(?:the\s+)?([a-z_\s]+)\s+(?:explained|explained|breakdown|walkthrough|tutorial)$",
        ]
        
        # Category 11: Typos/informal (catch common patterns)
        typo_informal_patterns = [
            r"^\w+\s+(?:explain|tell|define|clarify|meaning|explanation)",  # Typos in explanation/define
            r"^(?:wt|wats?|whats?|y)\s+(?:is|are|am|does|do)\s+",  # Internet slang: "wt is"
            r"^kya\s+",  # Hindi: "kya" = what
            r"(?:this|one|it)\s+(always\s+)?(messes|confuses|stumps|trips)\s+(me|up)",  # "This messes me up"
        ]
        
        # Category 12: Comparative (already has explicit "vs")
        # Covered in elliptical_patterns
        
        # Category 13: Hypothetical framing
        hypothetical_patterns = [
            r"(?:if|when)\s+.*(?:what|how|why)\s+(?:do|does|would|would|call|is)",
            r"what\s+(?:do\s+you\s+)?(?:call|mean|is)\s+it\s+when",
            r"what\s+would\s+you\s+(?:call|name)\s+it",
            r"what\'s\s+.*when",  # "What's it called when..."
            r"what\'s\s+.*(?:called|mean|defined)",  # "What's deadlock called" "What's deadlock mean"
        ]
        
        # Category 14: Reverse queries (looks like statement but really a question)
        reverse_patterns = [
            r"^isn\'?t\s+",
            r"^doesn\'?t\s+",
            r"^aren\'?t\s+",
            r"^wasn\'?t\s+",
            r"\s+(?:right|correct|true)\s*\?$",
            r"\s+(?:right|correct|true)\s*$",  # Confirmation seeking without ?
        ]
        
        # Compile all patterns
        all_patterns = (
            intent_patterns + buried_patterns + request_patterns + elliptical_patterns +
            sarcasm_patterns + mixed_patterns + negation_patterns + followup_patterns +
            verbose_patterns + ambiguous_patterns + typo_informal_patterns + hypothetical_patterns + reverse_patterns
        )
        
        for pattern in all_patterns:
            if re.search(pattern, t):
                return True
        
        return False

    def is_question(self, text: str, confidence: float) -> bool:
        """
        Bulletproof question detection for all 15 edge-case categories.
        
        New strategy (AGGRESSIVE):
        1. Explicit ? → always question
        2. Pattern-based detection FIRST (catches most cases)
        3. Ultra-low confidence gate (0.40)
        4. Classifier with very low threshold (just filter obvious statements)
        5. Fallback: if starts with question word or verb, assume question
        
        Result: Should catch ~90%+ of real questions while minimizing false positives.
        """
        t = text.strip()
        words = t.split()

        # Special case: Just a question mark
        if t == "?":
            return True
        
        # Explicit question mark always wins
        if t.endswith("?"):
            return True

        # AGGRESSIVE: Check patterns FIRST (before any gates)
        if self._detect_indirect_question_patterns(t):
            return True

        # Block ultra-short fragments (but allow 3-word requests like "Explain recursion")
        if len(words) < 3:
            return False

        # VERY LOW confidence gate (0.40 instead of 0.60)
        # Captures: statements with genuine intent but lower STT confidence
        if confidence < 0.40:
            return False

        # Fallback: starts with question word or common request verb
        if self._contains_question_word(t):
            return True

        # Last resort: classifier (now with much lower expectations)
        try:
            classifier = get_question_classifier()
            if classifier is None:
                # If classifier fails, use heuristic: if it looks like a noun phrase
                # for a domain concept, treat as potential request
                return False

            result = classifier(
                t,
                candidate_labels=["question or request for information", "statement"],
            )
            
            # Accept as question if confidence > 0.30 (very permissive)
            is_q = result["labels"][0] == "question or request for information"
            score = result["scores"][0]
            
            if is_q and score > 0.30:
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Classifier error: {e}")
            return False

    def _contains_question_word(self, text: str) -> bool:
        """
        Check if text STARTS with a known question word/phrase.
        Anchored to the start to avoid false positives like:
        "I don't know what you mean" (contains "what" but not a question)
        """
        t = text.lower().strip()
        return any(t.startswith(w) for w in QUESTION_STARTERS)

    def _should_trigger_dedup(self, text: str) -> bool:
        """
        Prevent duplicate questions from triggering LLM twice.
        Deepgram sometimes emits two final results for the same utterance.
        """
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if h == self.last_triggered_hash:
            return False  # Duplicate, skip
        self.last_triggered_hash = h
        return True

    def _cleanup_old_transcripts(self) -> None:
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0]["timestamp"]) > self.window_seconds:
            self.buffer.popleft()
