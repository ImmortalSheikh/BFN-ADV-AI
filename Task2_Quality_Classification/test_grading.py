"""
test_grading.py
Unit tests for the grading logic in grading.py.
Tests the A/B/C grade assignment and inventory action functions
to ensure they correctly implement the case study thresholds.
"""

import unittest
from grading import simulate_quality_scores, assign_grade, get_inventory_action


class TestAssignGrade(unittest.TestCase):
    """Tests for the assign_grade function."""

    def test_grade_a(self):
        """High quality scores should return Grade A."""
        self.assertEqual(assign_grade(80, 85, 75), "A")

    def test_grade_b(self):
        """Mid range scores should return Grade B."""
        self.assertEqual(assign_grade(68, 72, 62), "B")

    def test_grade_c(self):
        """Low scores should return Grade C."""
        self.assertEqual(assign_grade(50, 55, 45), "C")

    def test_grade_a_boundary(self):
        """Exactly at Grade A thresholds should return A."""
        self.assertEqual(assign_grade(75, 80, 70), "A")

    def test_grade_b_boundary(self):
        """Exactly at Grade B thresholds should return B."""
        self.assertEqual(assign_grade(65, 70, 60), "B")


class TestGetInventoryAction(unittest.TestCase):
    """Tests for the get_inventory_action function."""

    def test_grade_a_action(self):
        """Grade A should recommend full price stocking."""
        action = get_inventory_action("A")
        self.assertIn("full price", action)

    def test_grade_b_action(self):
        """Grade B should recommend a discount."""
        action = get_inventory_action("B")
        self.assertIn("discount", action)

    def test_grade_c_action(self):
        """Grade C should recommend removal or heavy discount."""
        action = get_inventory_action("C")
        self.assertIn("discount", action)


class TestSimulateQualityScores(unittest.TestCase):
    """Tests for the simulate_quality_scores function."""

    def test_healthy_scores_high(self):
        """Healthy produce with high confidence should have high scores."""
        color, size, ripeness = simulate_quality_scores(1.0, True)
        self.assertGreaterEqual(color, 75)
        self.assertGreaterEqual(size, 75)

    def test_rotten_scores_low(self):
        """Rotten produce should have lower quality scores."""
        color, size, ripeness = simulate_quality_scores(1.0, False)
        self.assertLess(color, 75)
        self.assertLess(ripeness, 75)


if __name__ == "__main__":
    unittest.main()