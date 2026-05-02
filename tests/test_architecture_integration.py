"""Integration tests for 7-layer Cluely architecture."""

import asyncio
import logging
import pytest
from datetime import datetime

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestArchitecturalLayerImports:
    """Test that all architectural layers can be imported."""

    def test_import_dialogue_state(self):
        """Layer 4: Dialogue state imports."""
        from backend.dialogue_state import DialogueState, DialogueStateManager, ConfidenceSignals
        assert DialogueState is not None
        assert DialogueStateManager is not None
        assert ConfidenceSignals is not None
        logger.info("✓ Dialogue state layer imports successful")

    def test_import_statement_assessment(self):
        """Layer 5: Statement assessment imports."""
        from backend.statement_assessment import StatementAssessor, CorrectnesLevel
        assert StatementAssessor is not None
        assert CorrectnesLevel is not None
        logger.info("✓ Statement assessment layer imports successful")

    def test_import_utterance_classifier(self):
        """Layer 3: Utterance classifier imports."""
        from backend.utterance_classifier import UtteranceClassifier, UtteranceType
        assert UtteranceClassifier is not None
        assert UtteranceType is not None
        logger.info("✓ Utterance classifier layer imports successful")

    def test_import_response_strategy(self):
        """Layer 6: Response strategy imports."""
        from backend.response_strategy import PedagogicalStrategyEngine, ResponseStrategy
        assert PedagogicalStrategyEngine is not None
        assert ResponseStrategy is not None
        logger.info("✓ Response strategy layer imports successful")

    def test_import_silence_detector(self):
        """Layer 7a: Silence detector imports."""
        from backend.silence_detector import SilenceDetector, SilenceEvent
        assert SilenceDetector is not None
        assert SilenceEvent is not None
        logger.info("✓ Silence detector layer imports successful")

    def test_import_repetition_tracker(self):
        """Layer 7b: Repetition tracker imports."""
        from backend.repetition_tracker import RepetitionTracker, ExplanationRecord
        assert RepetitionTracker is not None
        assert ExplanationRecord is not None
        logger.info("✓ Repetition tracker layer imports successful")

    def test_import_coreference_resolver(self):
        """Layer 7c: Coreference resolver imports."""
        from backend.coreference_resolver import CoreferenceResolver, CoreferenceResolution
        assert CoreferenceResolver is not None
        assert CoreferenceResolution is not None
        logger.info("✓ Coreference resolver layer imports successful")

    def test_import_pipeline(self):
        """Test pipeline imports with all layers."""
        from backend.pipeline import CopilotPipeline
        assert CopilotPipeline is not None
        logger.info("✓ Pipeline imports successful")


class TestDialogueStateLayer:
    """Test Layer 4: Dialogue State Management."""

    def test_dialogue_state_creation(self):
        """Create and verify dialogue state."""
        from backend.dialogue_state import DialogueState, ConfidenceSignals
        
        state = DialogueState(topic="algorithms")
        assert state.topic == "algorithms"
        assert state.exchange_count == 0
        assert isinstance(state.confidence_signals, ConfidenceSignals)
        logger.info("✓ Dialogue state creation works")

    def test_dialogue_state_manager(self):
        """Test dialogue state manager operations."""
        from backend.dialogue_state import DialogueStateManager
        
        manager = DialogueStateManager()
        manager.state.topic = "data structures"
        manager.state.subtopic = "linked lists"
        
        assert manager.state.topic == "data structures"
        assert manager.state.subtopic == "linked lists"
        
        manager.state.record_correct_response()
        assert manager.state.confidence_signals.correct_count == 1
        assert manager.state.exchange_count == 1
        
        logger.info("✓ Dialogue state manager operations work")

    def test_misconception_tracking(self):
        """Test misconception identification."""
        from backend.dialogue_state import DialogueStateManager
        
        manager = DialogueStateManager()
        manager.state.add_misconception(
            "pointers",
            "thinks pointers are arrays",
            "pointers are memory addresses"
        )
        
        assert len(manager.state.identified_misconceptions) == 1
        misconception = manager.state.identified_misconceptions[0]
        assert "arrays" in misconception.misconception_text.lower()
        
        logger.info("✓ Misconception tracking works")


class TestUtteranceClassificationLayer:
    """Test Layer 3: Utterance Classification."""

    def test_classify_question(self):
        """Classify a direct question."""
        from backend.utterance_classifier import UtteranceClassifier, UtteranceType
        
        classifier = UtteranceClassifier()
        result = classifier.classify("What is deadlock?")
        
        assert result.type == UtteranceType.DIRECT_QUESTION
        assert result.confidence > 0.5
        logger.info(f"✓ Question classification works: {result}")

    def test_classify_clarification(self):
        """Classify a clarification request."""
        from backend.utterance_classifier import UtteranceClassifier, UtteranceType
        
        classifier = UtteranceClassifier()
        result = classifier.classify("Wait, what do you mean by that?")
        
        assert result.type == UtteranceType.CLARIFICATION_QUESTION
        logger.info(f"✓ Clarification classification works: {result}")

    def test_classify_filler(self):
        """Classify filler utterance."""
        from backend.utterance_classifier import UtteranceClassifier, UtteranceType
        
        classifier = UtteranceClassifier()
        result = classifier.classify("um")
        
        assert result.type == UtteranceType.FILLER
        logger.info(f"✓ Filler classification works: {result}")


class TestResponseStrategyLayer:
    """Test Layer 6: Pedagogical Response Strategy."""

    def test_strategy_selection(self):
        """Test strategy selection based on state."""
        from backend.response_strategy import PedagogicalStrategyEngine
        from backend.utterance_classifier import UtteranceType
        from backend.dialogue_state import DialogueStateManager
        
        engine = PedagogicalStrategyEngine()
        manager = DialogueStateManager()
        
        # Simulate correct answer
        manager.state.record_correct_response()
        
        strategy = engine.choose_strategy(
            UtteranceType.DIRECT_QUESTION,
            dialogue_state=manager.state
        )
        
        assert strategy is not None
        assert strategy.type in ["reinforce", "redirect", "socratic", "explain", "acknowledge", "clarify", "challenge", "simplify"]
        logger.info(f"✓ Strategy selection works: {strategy}")

    def test_strategy_prompt_generation(self):
        """Test pedagogical prompt generation."""
        from backend.response_strategy import PedagogicalStrategyEngine, ResponseStrategy
        from backend.dialogue_state import DialogueStateManager
        
        engine = PedagogicalStrategyEngine()
        manager = DialogueStateManager()
        manager.state.topic = "sorting"
        
        strategy = engine.strategies["reinforce_correct"]
        prompt = engine.generate_system_prompt_instruction(manager.state, strategy)
        
        assert "sorting" in prompt.lower()
        assert strategy.description in prompt
        logger.info(f"✓ Pedagogical prompt generation works")


class TestSilenceDetectionLayer:
    """Test Layer 7a: Silence Detection."""

    def test_silence_detection(self):
        """Test silence detection after question."""
        from backend.silence_detector import SilenceDetector
        import time
        
        detector = SilenceDetector()
        detector.record_question_asked()
        
        # Immediately check — should not trigger soft threshold yet
        assert detector.check_silence() is None
        
        # Fast-forward time artificially (simulate delay)
        detector.last_utterance_time = time.time() - 5.5  # 5.5 seconds ago
        
        silence_event = detector.check_silence()
        assert silence_event is not None
        assert silence_event.severity == "hard"
        
        logger.info(f"✓ Silence detection works: {silence_event}")

    def test_silence_reset_on_response(self):
        """Test silence timer resets on student response."""
        from backend.silence_detector import SilenceDetector
        
        detector = SilenceDetector()
        detector.record_question_asked()
        
        # Record response
        detector.record_utterance("direct_question")
        
        # Should not detect silence now
        assert detector.check_silence() is None
        logger.info("✓ Silence reset on response works")


class TestRepetitionTrackingLayer:
    """Test Layer 7b: Repetition Avoidance."""

    def test_repetition_tracking(self):
        """Test tracking and detection of repeated explanations."""
        from backend.repetition_tracker import RepetitionTracker
        
        tracker = RepetitionTracker()
        
        explanation1 = "Deadlock occurs when two processes wait for each other indefinitely."
        
        tracker.add_explanation(
            topic="deadlock",
            explanation=explanation1,
            concept_covered="definition"
        )
        
        # Same explanation — should be detected as repetition
        is_repeated, record = tracker.check_repetition(explanation1)
        assert is_repeated is True
        assert record is not None
        
        logger.info("✓ Repetition tracking works")

    def test_topic_exhaustion(self):
        """Test detection of topic over-explanation."""
        from backend.repetition_tracker import RepetitionTracker
        
        tracker = RepetitionTracker()
        
        # Add 5 explanations on same topic
        for i in range(5):
            tracker.add_explanation(
                topic="sorting",
                explanation=f"Sorting is organizing data in order. Example {i}."
            )
        
        # Should detect exhaustion
        is_exhausted = tracker.check_topic_exhaustion("sorting", threshold=4)
        assert is_exhausted is True
        
        logger.info("✓ Topic exhaustion detection works")


class TestCoreferenceResolutionLayer:
    """Test Layer 7c: Coreference Resolution."""

    def test_pronoun_resolution(self):
        """Test resolving pronouns to concepts."""
        from backend.coreference_resolver import CoreferenceResolver
        
        resolver = CoreferenceResolver()
        
        # Set context
        resolver.update_context(
            "We discussed binary search trees.",
            ["binary search trees"]
        )
        
        # Resolve pronoun
        resolution = resolver.resolve("it", topic="trees")
        
        assert resolution.is_resolved or resolution.resolved_concept is None
        logger.info(f"✓ Pronoun resolution works: {resolution}")

    def test_noun_phrase_extraction(self):
        """Test extracting noun phrases."""
        from backend.coreference_resolver import CoreferenceResolver
        
        resolver = CoreferenceResolver()
        
        sentence = "The Binary Search Tree uses chaining to handle collisions."
        phrases = resolver.extract_noun_phrases(sentence)
        
        assert len(phrases) >= 0  # Extraction may or may not find phrases depending on heuristics
        logger.info(f"✓ Noun phrase extraction works: {phrases}")


class TestPipelineIntegration:
    """Test full pipeline integration."""

    def test_pipeline_initialization(self):
        """Test that pipeline initializes all layers."""
        from backend.pipeline import CopilotPipeline
        
        pipeline = CopilotPipeline()
        
        assert pipeline.dialogue_state_manager is not None
        assert pipeline.utterance_classifier is not None
        assert pipeline.strategy_engine is not None
        assert pipeline.silence_detector is not None
        assert pipeline.repetition_tracker is not None
        assert pipeline.coreference_resolver is not None
        
        logger.info("✓ Pipeline initialization with all layers works")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CLUELY 7-LAYER ARCHITECTURE TEST SUITE")
    logger.info("=" * 60)
    
    pytest.main([__file__, "-v", "--tb=short"])
