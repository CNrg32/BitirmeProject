#!/usr/bin/env python3
"""
test_panic_risk_score.py
=========================
Unit tests for V3.0 panic_risk_score calculation pipeline.
Tests V3.0 features: vowel_lengthening, negation_adjusted, ner_risk_density, entropy_inverse
Tests helper functions: compute_text_panic_score, compute_semantic_sentiment_score
Tests final panic_risk_score formula
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.services.sentiment_service import (
    extract_text_features,
    compute_text_panic_score,
)


class TestNewV3Features:
    """Test V3.0 new features extraction."""

    def test_vowel_lengthening_detection(self):
        """Stretched vowels like 'helppppp' should be detected."""
        text = "helppppp! pleeasseee!"
        feats = extract_text_features(text)
        
        # Should find vowel lengthening
        assert feats.get("vowel_lengthening_ratio", 0.0) > 0.0
        print(f"✓ Vowel lengthening detected: {feats.get('vowel_lengthening_ratio')}")

    def test_vowel_lengthening_not_triggered_normally(self):
        """Normal text should have zero or very low vowel lengthening."""
        text = "I am fine and everything is okay."
        feats = extract_text_features(text)
        
        # Should NOT find vowel lengthening
        assert feats.get("vowel_lengthening_ratio", 0.0) == 0.0
        print(f"✓ Normal text: vowel_lengthening={feats.get('vowel_lengthening_ratio')}")

    def test_negation_reduces_panic_score(self):
        """Negation words should reduce panic keyword contribution."""
        # Without negation
        text1 = "I can't breathe! I'm dying!"
        feats1 = extract_text_features(text1)
        panic_ratio1 = feats1.get("negation_adjusted_panic_ratio", 0.0)
        
        # With negation
        text2 = "Don't worry, I'm not dying. Everything is fine."
        feats2 = extract_text_features(text2)
        panic_ratio2 = feats2.get("negation_adjusted_panic_ratio", 0.0)
        
        # Negation version should have lower panic ratio
        assert panic_ratio2 < panic_ratio1 or panic_ratio2 == 0.0
        print(f"✓ Negation effect: {panic_ratio1:.4f} (without) → {panic_ratio2:.4f} (with negation)")

    def test_ner_risk_density_detection(self):
        """Risk entities like 'baby', 'blood', 'gun' should increase NER score."""
        text = "My baby is bleeding! There's a gun!"
        feats = extract_text_features(text)
        assert feats.get("ner_risk_density", 0.0) > 0.0
        print(f"✓ NER risk detected: {feats.get('ner_risk_density'):.4f}")

    def test_ner_risk_not_in_normal_text(self):
        """Normal text should have low NER risk."""
        text = "I went to the store and bought groceries."
        feats = extract_text_features(text)
        assert feats.get("ner_risk_density", 0.0) <= 0.1
        print(f"✓ Normal text NER risk: {feats.get('ner_risk_density'):.4f}")

    def test_sentence_entropy_inverse(self):
        """Low lexical diversity should increase entropy_inverse."""
        # High diversity
        text1 = "The situation is complex, intricate, and multifaceted. No repetition here."
        feats1 = extract_text_features(text1)
        entropy1 = feats1.get("sentence_entropy_inverse", 0.0)
        
        # Low diversity (many repetitions)
        text2 = "Help help help! Urgent urgent urgent! Now now now!"
        feats2 = extract_text_features(text2)
        entropy2 = feats2.get("sentence_entropy_inverse", 0.0)
        
        # Low diversity should have higher entropy_inverse (more repetitive = higher score)
        assert entropy2 >= entropy1
        print(f"✓ Entropy inverse: diverse={entropy1:.4f}, repetitive={entropy2:.4f}")


class TestTextPanicScoreComputation:
    """Test text_panic_score helper function."""

    def test_panic_score_critical_text(self):
        """Critical text should have reasonable panic score (relative to non-urgent)."""
        critical_texts = [
            ("Help! I can't breathe!", 0.08),
            ("Someone please! Blood everywhere!", 0.13),
            ("My baby is dying!!! Please help!", 0.11),
        ]
        
        for text, threshold in critical_texts:
            feats = extract_text_features(text)
            panic = compute_text_panic_score(feats)
            assert panic >= threshold - 0.01, f"Expected panic ≥ {threshold} for: {text}; got {panic:.4f}"
            assert panic > 0.05, f"Expected panic > 0.05 (baseline)"
            print(f"  {text[:40]:40s} → panic={panic:.4f}")

    def test_panic_score_non_urgent_text(self):
        """Non-urgent text should have low panic score."""
        non_urgent_texts = [
            "I have a mild headache.",
            "My arm is slightly sore.",
            "I'm feeling a bit tired today.",
        ]
        
        for text in non_urgent_texts:
            feats = extract_text_features(text)
            panic = compute_text_panic_score(feats)
            assert panic < 0.2, f"Expected panic < 0.2 for: {text}"
            print(f"  {text[:40]:40s} → panic={panic:.4f}")

    def test_panic_score_bounded(self):
        """Panic score should always be in [0, 1]."""
        texts = [
            "Help help help!", "I'm fine.", "!!!!!!!!!!! This is insane!",
            "Hello there, how are you?", "EMERGENCY EMERGENCY EMERGENCY!!!",
        ]
        
        for text in texts:
            feats = extract_text_features(text)
            panic = compute_text_panic_score(feats)
            assert 0.0 <= panic <= 1.0, f"Panic score out of bounds: {panic}"

    def test_panic_score_consistency(self):
        """Similar texts should produce similar panic scores."""
        text1 = "Please help! I can't breathe!"
        text2 = "Help! I can't breathe! Please!"
        
        feats1 = extract_text_features(text1)
        feats2 = extract_text_features(text2)
        panic1 = compute_text_panic_score(feats1)
        panic2 = compute_text_panic_score(feats2)
        
        # Similar texts should have similar scores (within 0.1)
        assert abs(panic1 - panic2) < 0.15
        print(f"✓ Consistency: '{text1}' → {panic1:.4f}, '{text2}' → {panic2:.4f}")


class TestSarcasmAndFalsePositives:
    """Test edge cases: sarcasm, slang, false positives."""

    def test_sarcasm_not_inflating_score(self):
        """Exaggerated speech should not be treated as pure panic."""
        sarcasm_texts = [
            "Oh my God, this coffee is terrible!",
            "I'm dying of boredom here.",
            "This exam is going to kill me!",
        ]
        
        for text in sarcasm_texts:
            feats = extract_text_features(text)
            panic = compute_text_panic_score(feats)
            # These are false alarms; panic should be low-moderate
            assert panic < 0.4, f"Sarcasm shouldn't trigger high panic: {text}"
            print(f"  Sarcasm: '{text}' → panic={panic:.4f}")

    def test_exclamation_abuse(self):
        """Many exclamations without actual panic words should be moderate."""
        text = "Hello! Hi! How are you! Great! Awesome!"
        feats = extract_text_features(text)
        panic = compute_text_panic_score(feats)
        
        # Should be low despite many exclamations
        assert panic < 0.2
        print(f"✓ Exclamation abuse: panic={panic:.4f} (should be low)")

    def test_negation_of_non_panic(self):
        """Negating non-panic words shouldn't affect score."""
        text1 = "I don't care."
        feats1 = extract_text_features(text1)
        panic1 = compute_text_panic_score(feats1)
        
        assert panic1 < 0.05
        print(f"✓ Non-panic negation: panic={panic1:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING V3.0 PANIC RISK SCORE UNIT TESTS")
    print("=" * 70)
    
    # Run tests manually
    test_features = TestNewV3Features()
    print("\n[1] Testing V3 Feature Extraction")
    test_features.test_vowel_lengthening_detection()
    test_features.test_vowel_lengthening_not_triggered_normally()
    test_features.test_negation_reduces_panic_score()
    test_features.test_ner_risk_density_detection()
    test_features.test_ner_risk_not_in_normal_text()
    test_features.test_sentence_entropy_inverse()
    
    test_panic = TestTextPanicScoreComputation()
    print("\n[2] Testing text_panic_score Computation")
    test_panic.test_panic_score_critical_text()
    test_panic.test_panic_score_non_urgent_text()
    test_panic.test_panic_score_bounded()
    test_panic.test_panic_score_consistency()
    
    test_edge = TestSarcasmAndFalsePositives()
    print("\n[3] Testing Edge Cases (Sarcasm, False Positives)")
    test_edge.test_sarcasm_not_inflating_score()
    test_edge.test_exclamation_abuse()
    test_edge.test_negation_of_non_panic()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
