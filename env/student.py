"""
Student Knowledge Model — Bayesian Knowledge Tracing (BKT)
Each concept has a hidden mastery probability that evolves with practice.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


CONCEPTS = [
    "algebra_basics",
    "linear_equations",
    "quadratic_equations",
    "functions",
    "trigonometry",
    "probability",
    "statistics",
    "calculus_intro",
    "geometry",
    "number_theory",
]

# BKT parameters per concept: (p_init, p_learn, p_forget, p_slip, p_guess)
CONCEPT_PARAMS: Dict[str, Dict[str, float]] = {
    "algebra_basics":       {"p_init": 0.4,  "p_learn": 0.3,  "p_forget": 0.05, "p_slip": 0.1,  "p_guess": 0.2},
    "linear_equations":     {"p_init": 0.3,  "p_learn": 0.25, "p_forget": 0.05, "p_slip": 0.12, "p_guess": 0.18},
    "quadratic_equations":  {"p_init": 0.2,  "p_learn": 0.2,  "p_forget": 0.06, "p_slip": 0.15, "p_guess": 0.15},
    "functions":            {"p_init": 0.25, "p_learn": 0.22, "p_forget": 0.05, "p_slip": 0.13, "p_guess": 0.16},
    "trigonometry":         {"p_init": 0.15, "p_learn": 0.18, "p_forget": 0.07, "p_slip": 0.18, "p_guess": 0.12},
    "probability":          {"p_init": 0.3,  "p_learn": 0.25, "p_forget": 0.06, "p_slip": 0.12, "p_guess": 0.2},
    "statistics":           {"p_init": 0.28, "p_learn": 0.23, "p_forget": 0.05, "p_slip": 0.11, "p_guess": 0.18},
    "calculus_intro":       {"p_init": 0.1,  "p_learn": 0.15, "p_forget": 0.08, "p_slip": 0.2,  "p_guess": 0.1},
    "geometry":             {"p_init": 0.35, "p_learn": 0.28, "p_forget": 0.05, "p_slip": 0.1,  "p_guess": 0.2},
    "number_theory":        {"p_init": 0.2,  "p_learn": 0.2,  "p_forget": 0.06, "p_slip": 0.15, "p_guess": 0.14},
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Prerequisite graph: concept -> [prerequisites]
PREREQUISITES: Dict[str, List[str]] = {
    "algebra_basics":       [],
    "linear_equations":     ["algebra_basics"],
    "quadratic_equations":  ["algebra_basics", "linear_equations"],
    "functions":            ["algebra_basics", "linear_equations"],
    "trigonometry":         ["functions", "geometry"],
    "probability":          ["algebra_basics"],
    "statistics":           ["algebra_basics", "probability"],
    "calculus_intro":       ["functions", "trigonometry"],
    "geometry":             ["algebra_basics"],
    "number_theory":        ["algebra_basics"],
}


@dataclass
class StudentState:
    """Full hidden + observable state of a simulated student."""
    # Hidden: true mastery probability per concept
    mastery: Dict[str, float] = field(default_factory=dict)
    # Observable: number of attempts per concept per difficulty
    attempts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Observable: correct answer count
    correct: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Engagement score [0,1] — drops if questions are too easy/hard
    engagement: float = 1.0
    # Fatigue accumulates per step
    fatigue: float = 0.0
    # Session step count
    step_count: int = 0
    # Streak per concept
    streak: Dict[str, int] = field(default_factory=dict)

    def to_obs_vector(self, concepts: List[str]) -> np.ndarray:
        """Return observable features as a flat numpy array."""
        obs = []
        for c in concepts:
            total_att = sum(self.attempts.get(c, {d: 0 for d in DIFFICULTY_LEVELS}).values())
            total_cor = sum(self.correct.get(c, {d: 0 for d in DIFFICULTY_LEVELS}).values())
            success_rate = total_cor / max(total_att, 1)
            obs.extend([
                success_rate,
                min(total_att / 20.0, 1.0),   # normalised attempts
                self.streak.get(c, 0) / 5.0,   # normalised streak
            ])
        obs.append(self.engagement)
        obs.append(self.fatigue)
        return np.array(obs, dtype=np.float32)


class StudentSimulator:
    """
    Simulates a student responding to questions using Bayesian Knowledge Tracing.
    The agent never sees `mastery` directly — it must infer from responses.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state: StudentState = self._init_state()

    def _init_state(self) -> StudentState:
        mastery = {}
        for c, p in CONCEPT_PARAMS.items():
            # Sample initial mastery with some variance
            mastery[c] = float(np.clip(
                self.rng.normal(p["p_init"], 0.05), 0.05, 0.95
            ))
        attempts = {c: {d: 0 for d in DIFFICULTY_LEVELS} for c in CONCEPTS}
        correct  = {c: {d: 0 for d in DIFFICULTY_LEVELS} for c in CONCEPTS}
        streak   = {c: 0 for c in CONCEPTS}
        return StudentState(mastery=mastery, attempts=attempts,
                            correct=correct, streak=streak)

    def reset(self, seed: Optional[int] = None) -> StudentState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._init_state()
        return self.state

    def answer_question(self, concept: str, difficulty: str) -> bool:
        """
        Simulate student answering a question.
        Returns True if correct.
        Updates internal BKT state.
        """
        assert concept in CONCEPTS, f"Unknown concept: {concept}"
        assert difficulty in DIFFICULTY_LEVELS, f"Unknown difficulty: {difficulty}"

        p = CONCEPT_PARAMS[concept]
        m = self.state.mastery[concept]

        # Difficulty modifiers on slip/guess
        diff_mod = {"easy": -0.05, "medium": 0.0, "hard": 0.08}[difficulty]
        p_slip  = np.clip(p["p_slip"]  + diff_mod, 0.01, 0.4)
        p_guess = np.clip(p["p_guess"] - diff_mod, 0.01, 0.4)

        # P(correct) = P(mastered)*(1-slip) + P(not mastered)*guess
        p_correct = m * (1 - p_slip) + (1 - m) * p_guess
        correct = bool(self.rng.random() < p_correct)

        # BKT update: posterior mastery given response
        if correct:
            p_mastered_given_correct = (m * (1 - p_slip)) / max(p_correct, 1e-9)
        else:
            p_mastered_given_incorrect = (m * p_slip) / max(1 - p_correct, 1e-9)
            p_mastered_given_correct = p_mastered_given_incorrect  # rename for clarity

        # Learning update
        new_m = p_mastered_given_correct + (1 - p_mastered_given_correct) * p["p_learn"]
        # Forgetting
        new_m = new_m * (1 - p["p_forget"])
        self.state.mastery[concept] = float(np.clip(new_m, 0.0, 1.0))

        # Track attempts / correct
        self.state.attempts[concept][difficulty] += 1
        if correct:
            self.state.correct[concept][difficulty] += 1
            self.state.streak[concept] = self.state.streak.get(concept, 0) + 1
        else:
            self.state.streak[concept] = 0

        # Engagement: reward zone-of-proximal-development, penalise misfit
        ideal_difficulty_score = self._ideal_difficulty_score(concept, difficulty)
        self.state.engagement = float(np.clip(
            self.state.engagement * 0.95 + 0.05 * ideal_difficulty_score, 0.1, 1.0
        ))

        # Fatigue increases per step
        self.state.fatigue = min(self.state.fatigue + 0.02, 1.0)
        self.state.step_count += 1

        return correct

    def _ideal_difficulty_score(self, concept: str, difficulty: str) -> float:
        """
        Score 1.0 if difficulty matches student's current mastery zone.
        Zone of Proximal Development: difficulty should be just above mastery.
        """
        m = self.state.mastery[concept]
        target = {"easy": 0.3, "medium": 0.55, "hard": 0.8}[difficulty]
        return float(np.exp(-4 * (m - target) ** 2))

    def get_mastery_summary(self) -> Dict[str, float]:
        return dict(self.state.mastery)

    def prerequisite_readiness(self, concept: str) -> float:
        """Average mastery of prerequisites. 1.0 if no prerequisites."""
        prereqs = PREREQUISITES.get(concept, [])
        if not prereqs:
            return 1.0
        return float(np.mean([self.state.mastery[p] for p in prereqs]))
