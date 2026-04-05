"""
Baseline Agents — reproducible reference implementations.

Three agents of increasing sophistication:
1. RandomAgent        — random concept + difficulty
2. HeuristicAgent     — rule-based: pick weakest concept, ZPD difficulty
3. GreedyMasteryAgent — greedy: follow prerequisite order, adaptive difficulty
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from env.student import CONCEPTS, DIFFICULTY_LEVELS, PREREQUISITES


# ─────────────────────────────────────────────
# 1. Random Agent (lower bound)
# ─────────────────────────────────────────────

class RandomAgent:
    """Selects concept and difficulty uniformly at random."""
    name = "RandomAgent"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: Dict) -> Dict:
        return {
            "concept":    self.rng.choice(CONCEPTS),
            "difficulty": self.rng.choice(DIFFICULTY_LEVELS),
            "hint_given": False,
        }


# ─────────────────────────────────────────────
# 2. Heuristic Agent (mid baseline)
# ─────────────────────────────────────────────

class HeuristicAgent:
    """
    Rule-based agent:
    - Picks concept with lowest success rate among targets
    - Picks difficulty in zone of proximal development:
        success_rate < 0.4  → easy
        success_rate 0.4–0.7 → medium
        success_rate > 0.7   → hard
    - Avoids concepts whose prerequisites are not ready (readiness < 0.4)
    """
    name = "HeuristicAgent"

    def __init__(self, target_concepts: Optional[List[str]] = None):
        self.targets = target_concepts or CONCEPTS

    def __call__(self, obs: Dict) -> Dict:
        rates    = obs["concept_success_rates"]
        prereq_r = obs["prerequisite_readiness"]

        # Filter to concepts with ready prerequisites
        eligible = [
            c for c in self.targets
            if prereq_r.get(c, 1.0) >= 0.35
        ]
        if not eligible:
            eligible = self.targets  # fallback

        # Pick weakest concept
        concept = min(eligible, key=lambda c: rates.get(c, 0.0))

        # ZPD difficulty
        sr = rates.get(concept, 0.0)
        if sr < 0.40:
            difficulty = "easy"
        elif sr < 0.70:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return {"concept": concept, "difficulty": difficulty, "hint_given": False}


# ─────────────────────────────────────────────
# 3. Greedy Mastery Agent (strong baseline)
# ─────────────────────────────────────────────

class GreedyMasteryAgent:
    """
    Strongest baseline:
    - Scores each concept by (mastery_gap * prereq_readiness) to prioritise
    - Tracks estimated mastery via attempt-weighted success rate
    - Adjusts difficulty based on per-concept success rate + streak
    - Manages fatigue: drops to easy when fatigue > 0.65
    - Spreads budget across all target concepts proportionally
    """
    name = "GreedyMasteryAgent"

    def __init__(
        self,
        target_concepts: Optional[List[str]] = None,
        mastery_threshold: float = 0.70,
    ):
        self.targets   = target_concepts or CONCEPTS
        self.threshold = mastery_threshold
        self._topo_order = self._topo_sort(self.targets)

    def _topo_sort(self, concepts: List[str]) -> List[str]:
        """Sort concepts by prerequisite depth (shallow first)."""
        def depth(c: str, visited: frozenset = frozenset()) -> int:
            if c in visited:
                return 0
            prereqs = [p for p in PREREQUISITES.get(c, []) if p in concepts]
            if not prereqs:
                return 0
            return 1 + max(depth(p, visited | {c}) for p in prereqs)
        return sorted(concepts, key=depth)

    def __call__(self, obs: Dict) -> Dict:
        rates    = obs["concept_success_rates"]
        attempts = obs["concept_attempt_counts"]
        streaks  = obs["concept_streaks"]
        fatigue  = obs["fatigue"]
        prereq_r = obs["prerequisite_readiness"]

        # Score each target concept:
        # priority = mastery_gap * prereq_readiness * depth_bonus
        best_concept = None
        best_score   = -1.0

        for i, c in enumerate(self._topo_order):
            sr          = rates.get(c, 0.0)
            att         = attempts.get(c, 0)
            gap         = max(self.threshold - sr, 0.0)
            prereq      = prereq_r.get(c, 1.0)

            # Hard skip: prerequisites not ready enough to attempt
            if prereq < 0.45 and att == 0:
                continue

            # Prereq readiness is a strong quadratic multiplier
            prereq_weight = prereq ** 2

            # Unvisited concepts get moderate novelty bump
            novelty = 1.2 if att == 0 else 1.0

            # Shallow concepts (earlier in topo) get priority
            depth_bonus = 1.0 + 0.08 * (len(self._topo_order) - i)

            score = gap * prereq_weight * novelty * depth_bonus

            if score > best_score:
                best_score   = score
                best_concept = c

        # Fallback: lowest success rate among targets
        if best_concept is None:
            best_concept = min(
                self.targets,
                key=lambda c: rates.get(c, 0.0)
            )

        concept = best_concept
        sr      = rates.get(concept, 0.0)
        streak  = streaks.get(concept, 0)

        # Difficulty selection
        if fatigue > 0.65:
            difficulty = "easy"
        elif streak >= 3 or sr > 0.70:
            difficulty = "hard"
        elif sr > 0.40:
            difficulty = "medium"
        else:
            difficulty = "easy"

        return {"concept": concept, "difficulty": difficulty, "hint_given": False}


# ─────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────

def get_agent(name: str, **kwargs):
    registry = {
        "random":   RandomAgent,
        "heuristic": HeuristicAgent,
        "greedy":   GreedyMasteryAgent,
    }
    if name not in registry:
        raise ValueError(f"Unknown agent '{name}'. Choose from {list(registry)}")
    return registry[name](**kwargs)
