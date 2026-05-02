"""
Comprehensive test suite for robust question detection.
Tests all 15 edge-case categories from stress-test.

Run with: python -m pytest tests/test_question_detection_comprehensive.py -v
Or standalone: python tests/test_question_detection_comprehensive.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.context import ContextEngine


def test_category_1_indirect_softened_questions():
    """Category 1: Indirect / Softened Questions - No explicit what/why/how"""
    engine = ContextEngine()
    
    test_cases = [
        ("I was always a bit confused about deadlocks", 0.65, True, "Confused about X"),
        ("Deadlock situations are kind of tricky to understand", 0.55, True, "Tricky to understand"),
        ("I never really got what deadlock actually means", 0.51, True, "Never got what means"),
        ("Deadlock… this one always messes with me", 0.60, True, "Messes with me"),
        ("I never really got what a deadlock means", 0.51, True, "Intent to understand"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_2_embedded_in_conversation():
    """Category 2: Embedded in Conversation - Buried inside longer sentence"""
    engine = ContextEngine()
    
    test_cases = [
        ("Yeah so we were discussing OS concepts and someone mentioned deadlock and I didn't really follow what that is", 0.68, True, "Embedded after context"),
        ("In interviews they ask about deadlock and I usually blank out on what it actually is", 0.72, True, "Embedded in workflow"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_3_command_disguised_as_statement():
    """Category 3: Command Disguised as Statement - Request without explicit verb"""
    engine = ContextEngine()
    
    test_cases = [
        ("Explain deadlock", 0.80, True, "Direct command"),
        ("Need an explanation of deadlock", 0.65, True, "Need explanation"),
        ("Deadlock explanation would help", 0.58, True, "Would help"),
        ("A quick deadlock explanation", 0.52, True, "Noun phrase request"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_4_elliptical_incomplete_queries():
    """Category 4: Elliptical / Incomplete Queries - Grammatically incomplete but semantically clear"""
    engine = ContextEngine()
    
    test_cases = [
        ("Deadlock?", 0.90, True, "Ultra-minimal question"),
        ("Deadlock in OS?", 0.88, True, "Minimal with context"),
        ("Deadlock meaning?", 0.85, True, "Query with noun"),
        ("Deadlock vs starvation?", 0.82, True, "Comparative query"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_5_sarcastic_rhetorical_tone():
    """Category 5: Sarcastic / Rhetorical Tone - Intent ≠ surface meaning"""
    engine = ContextEngine()
    
    test_cases = [
        ("Yeah because deadlock is sooo easy to understand right", 0.55, True, "Sarcasm with intent"),
        ("Oh sure deadlock totally makes sense to me", 0.50, True, "Mock agreement"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_6_multi_intent_mixed_signals():
    """Category 6: Multi-Intent / Mixed Signals - Part statement, part question"""
    engine = ContextEngine()
    
    test_cases = [
        ("Deadlock is when processes block each other right or am I missing something", 0.60, True, "Statement + confirmation"),
        ("I think deadlock happens due to resource contention but not fully sure", 0.55, True, "Explanation + doubt"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_7_negation_based_queries():
    """Category 7: Negation-Based Queries - Negation confuses intent detection"""
    engine = ContextEngine()
    
    test_cases = [
        ("Why isn't deadlock just starvation", 0.62, True, "Negation question"),
        ("Is deadlock not the same as livelock", 0.68, True, "Negation comparison"),
        ("Deadlock doesn't mean infinite waiting right", 0.55, True, "Negation confirmation"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_8_context_dependent_followups():
    """Category 8: Context-Dependent Follow-ups - Requires memory of previous turn"""
    engine = ContextEngine()
    
    test_cases = [
        ("And this happens because of what exactly", 0.70, True, "Why exactly"),
        ("So what does that mean in practice", 0.65, True, "Practical implication"),
        ("Why though", 0.55, True, "Ultra-minimal why"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_9_overly_verbose_noisy_inputs():
    """Category 9: Overly Verbose / Noisy Inputs - Signal buried in noise"""
    engine = ContextEngine()
    
    test_cases = [
        ("So basically I was going through OS concepts yesterday and like I saw this thing called deadlock and I kind of didn't get like what exactly is happening there", 0.48, True, "Verbose with buried intent"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_10_domain_ambiguous_queries():
    """Category 10: Domain-Ambiguous Queries - Could mean multiple things"""
    engine = ContextEngine()
    
    test_cases = [
        ("Deadlock meaning", 0.65, True, "Request for meaning"),
        ("Deadlock explanation", 0.70, True, "Request for explanation"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_11_typos_informal_language():
    """Category 11: Typos / Informal Language - String-based systems break"""
    engine = ContextEngine()
    
    test_cases = [
        ("deadlcok explain", 0.55, True, "Typo in word"),
        ("wt is deadlock", 0.60, True, "Informal abbreviation"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_12_comparative_without_explicit_ask():
    """Category 12: Comparative Without Explicit Ask - Implied request"""
    engine = ContextEngine()
    
    test_cases = [
        ("Deadlock vs livelock", 0.68, True, "Comparative no verb"),
        ("Deadlock compared to starvation", 0.65, True, "Explicit compared"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_13_hypothetical_framing():
    """Category 13: Hypothetical Framing - Hidden question"""
    engine = ContextEngine()
    
    test_cases = [
        ("If two processes wait on each other what's that called again", 0.72, True, "What's that called"),
        ("What do you call it when resources are held in a cycle", 0.68, True, "What do you call"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_14_reverse_queries_answer_to_ask():
    """Category 14: Reverse Queries (Answer → Ask) - Looks like confirmation"""
    engine = ContextEngine()
    
    test_cases = [
        ("Isn't deadlock when processes are stuck waiting forever", 0.70, True, "Reversed question"),
        ("Deadlock means circular wait right", 0.65, True, "Confirmation seeking"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_category_15_ultra_minimal_prompts():
    """Category 15: Ultra-Minimal Prompts - Zero structure"""
    engine = ContextEngine()
    
    test_cases = [
        ("Deadlock", 0.85, False, "Single word (no context)"),
        ("In OS: deadlock", 0.70, False, "Fragment with context"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def test_baseline_statements_should_not_trigger():
    """Baseline: Statements that should NOT trigger LLM"""
    engine = ContextEngine()
    
    test_cases = [
        ("Okay", 0.90, False, "Ack statement"),
        ("Got it", 0.85, False, "Confirmation"),
        ("Makes sense", 0.88, False, "Understanding statement"),
        ("Sounds good", 0.82, False, "Agreement"),
    ]
    
    results = []
    for text, conf, expected, desc in test_cases:
        result = engine.is_question(text, conf)
        status = "✅" if result == expected else "❌"
        results.append((status, result, expected, desc, text[:50]))
        print(f"  {status} {desc:30} conf={conf:.2f} → {result} (expect {expected})")
    
    return results


def run_all_tests():
    """Run all test categories and aggregate results."""
    
    categories = [
        ("Category 1: Indirect/Softened Questions", test_category_1_indirect_softened_questions),
        ("Category 2: Embedded in Conversation", test_category_2_embedded_in_conversation),
        ("Category 3: Command Disguised as Statement", test_category_3_command_disguised_as_statement),
        ("Category 4: Elliptical/Incomplete Queries", test_category_4_elliptical_incomplete_queries),
        ("Category 5: Sarcastic/Rhetorical Tone", test_category_5_sarcastic_rhetorical_tone),
        ("Category 6: Multi-Intent/Mixed Signals", test_category_6_multi_intent_mixed_signals),
        ("Category 7: Negation-Based Queries", test_category_7_negation_based_queries),
        ("Category 8: Context-Dependent Follow-ups", test_category_8_context_dependent_followups),
        ("Category 9: Overly Verbose/Noisy Inputs", test_category_9_overly_verbose_noisy_inputs),
        ("Category 10: Domain-Ambiguous Queries", test_category_10_domain_ambiguous_queries),
        ("Category 11: Typos/Informal Language", test_category_11_typos_informal_language),
        ("Category 12: Comparative Without Explicit Ask", test_category_12_comparative_without_explicit_ask),
        ("Category 13: Hypothetical Framing", test_category_13_hypothetical_framing),
        ("Category 14: Reverse Queries", test_category_14_reverse_queries_answer_to_ask),
        ("Category 15: Ultra-Minimal Prompts", test_category_15_ultra_minimal_prompts),
        ("Baseline: Statements (Should NOT trigger)", test_baseline_statements_should_not_trigger),
    ]
    
    all_results = []
    total_tests = 0
    total_passed = 0
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QUESTION DETECTION TEST SUITE")
    print("="*80 + "\n")
    
    for category_name, test_func in categories:
        print(f"\n{category_name}")
        print("-" * 80)
        
        results = test_func()
        all_results.extend(results)
        
        passed = sum(1 for r in results if r[0] == "✅")
        total = len(results)
        total_tests += total
        total_passed += passed
        
        print(f"  Summary: {passed}/{total} passed")
    
    # Final aggregate
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nTotal Tests: {total_tests}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_tests - total_passed}")
    print(f"Pass Rate: {100*total_passed/total_tests:.1f}%")
    print(f"Acceptable: {'✅ YES' if total_passed/total_tests >= 0.85 else '❌ NO (need ≥85% pass rate)'}")
    
    # Category breakdown
    print("\nCategory Breakdown:")
    category_totals = {}
    for status, result, expected, desc, text in all_results:
        # Extract category name from desc
        cat = desc.split(':')[0] if ':' in desc else desc
        if cat not in category_totals:
            category_totals[cat] = {"total": 0, "passed": 0}
        category_totals[cat]["total"] += 1
        if status == "✅":
            category_totals[cat]["passed"] += 1
    
    for cat in sorted(category_totals.keys()):
        p = category_totals[cat]["passed"]
        t = category_totals[cat]["total"]
        pct = 100*p/t
        status = "✅" if pct == 100 else "⚠️" if pct >= 80 else "❌"
        print(f"  {status} {cat:30} {p}/{t} ({pct:.0f}%)")


if __name__ == "__main__":
    run_all_tests()
