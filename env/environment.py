"""
AdaptiveTutorEnv — OpenEnv-compliant environment
Implements: reset() / step() / state() with typed Pydantic models.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from env.student import (
    StudentSimulator, StudentState,
    CONCEPTS, DIFFICULTY_LEVELS, PREREQUISITES
)


# ─────────────────────────────────────────────
# Typed Action & Observation Models (OpenEnv spec)
# ─────────────────────────────────────────────

class TutorAction(BaseModel):
    """Action: choose which concept and difficulty to present next."""
    concept: str = Field(..., description="One of the 10 math concepts")
    difficulty: str = Field(..., description="easy | medium | hard")
    hint_given: bool = Field(False, description="Whether to give a hint (costs engagement)")

    class Config:
        extra = "forbid"


class TutorObservation(BaseModel):
    """Observable state returned after each step."""
    # Per-concept observable stats (agent cannot see true mastery)
    concept_success_rates: Dict[str, float] = Field(
        ..., description="Success rate per concept [0,1]"
    )
    concept_attempt_counts: Dict[str, int] = Field(
        ..., description="Total attempts per concept"
    )
    concept_streaks: Dict[str, int] = Field(
        ..., description="Current correct streak per concept"
    )
    prerequisite_readiness: Dict[str, float] = Field(
        ..., description="Avg mastery of prerequisites per concept [0,1]"
    )
    engagement: float = Field(..., description="Student engagement [0,1]")
    fatigue: float = Field(..., description="Student fatigue [0,1]")
    step_count: int = Field(..., description="Steps taken this episode")
    last_correct: bool = Field(..., description="Was the last answer correct?")
    last_concept: str = Field(..., description="Last concept presented")
    last_difficulty: str = Field(..., description="Last difficulty presented")

    class Config:
        extra = "forbid"


class StepResult(BaseModel):
    """Full result of a step() call."""
    observation: TutorObservation
    reward: float = Field(..., description="Reward in [0,1]")
    done: bool = Field(..., description="Episode complete")
    info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class EpisodeInfo(BaseModel):
    """Returned by state() — full episode metadata."""
    task_id: str
    task_difficulty: str
    max_steps: int
    steps_remaining: int
    cumulative_reward: float
    observation: TutorObservation
    mastery_snapshot: Optional[Dict[str, float]] = None  # only exposed in eval mode

    class Config:
        extra = "forbid"


# ─────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict] = {
    "single_concept_mastery": {
        "difficulty": "easy",
        "description": "Bring one assigned concept from low to high mastery in 20 steps.",
        "max_steps": 20,
        "target_concepts": ["algebra_basics"],
        "mastery_threshold": 0.75,
        "score_fn": "single_concept",
    },
    "multi_concept_curriculum": {
        "difficulty": "medium",
        "description": (
            "Bring 5 concepts to mastery within 50 steps while maintaining "
            "student engagement above 0.5."
        ),
        "max_steps": 50,
        "target_concepts": [
            "algebra_basics", "linear_equations", "functions",
            "probability", "geometry"
        ],
        "mastery_threshold": 0.70,
        "score_fn": "multi_concept",
    },
    "exam_prep_sprint": {
        "difficulty": "hard",
        "description": (
            "Maximise predicted exam score across all 10 concepts within 80 steps, "
            "respecting prerequisite order and keeping fatigue below 0.8."
        ),
        "max_steps": 80,
        "target_concepts": CONCEPTS,
        "mastery_threshold": 0.65,
        "score_fn": "exam_prep",
    },
}


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────

class AdaptiveTutorEnv:
    """
    OpenEnv-compliant Adaptive Student Tutoring Environment.

    The agent selects which concept and difficulty to present.
    The student simulator responds probabilistically (BKT model).
    Reward is a composite of mastery gain, engagement, and efficiency.
    """

    VERSION = "1.0.0"
    ENV_ID  = "AdaptiveTutor-v1"

    def __init__(
        self,
        task_id: str = "single_concept_mastery",
        seed: int = 42,
        eval_mode: bool = False,
    ):
        assert task_id in TASK_REGISTRY, (
            f"Unknown task: {task_id}. Choose from {list(TASK_REGISTRY)}"
        )
        self.task_id    = task_id
        self.task_cfg   = TASK_REGISTRY[task_id]
        self.seed       = seed
        self.eval_mode  = eval_mode

        self._sim = StudentSimulator(seed=seed)
        self._step_count    = 0
        self._cum_reward    = 0.0
        self._prev_mastery: Dict[str, float] = {}
        self._last_obs: Optional[TutorObservation] = None
        self._last_correct  = False
        self._last_concept  = CONCEPTS[0]
        self._last_diff     = "easy"
        self._done          = False

    # ── OpenEnv API ──────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> TutorObservation:
        """Reset environment. Returns initial observation."""
        _seed = seed if seed is not None else self.seed
        self._sim.reset(seed=_seed)
        self._step_count  = 0
        self._cum_reward  = 0.0
        self._done        = False
        self._last_correct = False
        self._last_concept = self.task_cfg["target_concepts"][0]
        self._last_diff    = "easy"
        self._prev_mastery = dict(self._sim.state.mastery)
        obs = self._build_observation()
        self._last_obs = obs
        return obs

    def step(self, action: TutorAction | Dict) -> StepResult:
        """Execute one tutoring step. Returns StepResult."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        if isinstance(action, dict):
            action = TutorAction(**action)

        self._validate_action(action)

        # Student answers
        correct = self._sim.answer_question(action.concept, action.difficulty)
        self._last_correct = correct
        self._last_concept = action.concept
        self._last_diff    = action.difficulty
        self._step_count  += 1

        # Build reward
        reward = self._compute_reward(action, correct)
        self._cum_reward += reward

        # Check done
        done = self._check_done()
        self._done = done

        obs = self._build_observation()
        self._last_obs = obs
        self._prev_mastery = dict(self._sim.state.mastery)

        info = {
            "mastery_delta": {
                c: self._sim.state.mastery[c] - self._prev_mastery.get(c, 0)
                for c in CONCEPTS
            },
            "task_score": self._compute_task_score(),
            "step": self._step_count,
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> EpisodeInfo:
        """Return full episode state snapshot."""
        obs = self._last_obs or self._build_observation()
        return EpisodeInfo(
            task_id=self.task_id,
            task_difficulty=self.task_cfg["difficulty"],
            max_steps=self.task_cfg["max_steps"],
            steps_remaining=self.task_cfg["max_steps"] - self._step_count,
            cumulative_reward=self._cum_reward,
            observation=obs,
            mastery_snapshot=dict(self._sim.state.mastery) if self.eval_mode else None,
        )

    # ── Internal helpers ─────────────────────────────────

    def _validate_action(self, action: TutorAction):
        if action.concept not in CONCEPTS:
            raise ValueError(f"Invalid concept: {action.concept}")
        if action.difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty: {action.difficulty}")

    def _build_observation(self) -> TutorObservation:
        s = self._sim.state
        success_rates, attempt_counts, streaks = {}, {}, {}
        for c in CONCEPTS:
            total_att = sum(s.attempts[c].values())
            total_cor = sum(s.correct[c].values())
            success_rates[c]  = round(total_cor / max(total_att, 1), 4)
            attempt_counts[c] = total_att
            streaks[c]        = s.streak.get(c, 0)

        prereq_readiness = {
            c: round(self._sim.prerequisite_readiness(c), 4)
            for c in CONCEPTS
        }

        return TutorObservation(
            concept_success_rates=success_rates,
            concept_attempt_counts=attempt_counts,
            concept_streaks=streaks,
            prerequisite_readiness=prereq_readiness,
            engagement=round(s.engagement, 4),
            fatigue=round(s.fatigue, 4),
            step_count=self._step_count,
            last_correct=self._last_correct,
            last_concept=self._last_concept,
            last_difficulty=self._last_diff,
        )

    def _compute_reward(self, action: TutorAction, correct: bool) -> float:
        """
        Composite reward with partial progress signals:
        1. Mastery gain on target concepts (3x weight, primary signal)
        2. Terminal bonus at episode end  (shapes long-horizon credit)
        3. Over-drilling penalty          (discourages re-drilling mastered topics)
        4. Engagement preservation
        5. Prerequisite alignment bonus
        6. Zone of proximal development bonus
        7. Correct answer base reward
        8. Fatigue + hint penalties
        """
        concept = action.concept
        prev_m  = self._prev_mastery.get(concept, 0.0)
        curr_m  = self._sim.state.mastery[concept]
        mastery_gain = max(curr_m - prev_m, 0.0)

        target    = self.task_cfg["target_concepts"]
        threshold = self.task_cfg["mastery_threshold"]

        # 3x reward for mastery gains on unmastered target concepts
        already_mastered  = prev_m >= threshold
        target_multiplier = 0.3 if concept not in target else (
            0.5 if already_mastered else 3.0
        )

        # Penalise wasting steps on already-mastered concepts
        overdrill_penalty = 0.08 if (concept in target and already_mastered) else 0.0

        # Engagement component
        eng_reward = self._sim.state.engagement * 0.08

        # Prerequisite alignment bonus
        prereq_score = self._sim.prerequisite_readiness(concept)
        prereq_bonus = 0.05 * prereq_score

        # Zone of proximal development bonus
        zpd_score = self._sim._ideal_difficulty_score(concept, action.difficulty)
        zpd_bonus = 0.08 * zpd_score

        # Correct answer base reward
        correct_reward = 0.10 if correct else 0.0

        # Fatigue penalty
        fatigue_penalty = self._sim.state.fatigue * 0.04

        # Hint penalty
        hint_penalty = 0.05 if action.hint_given else 0.0

        # Terminal bonus: strong signal at episode end for PPO credit assignment
        terminal_bonus = 0.0
        if self._step_count >= self.task_cfg["max_steps"] - 1:
            terminal_bonus = self._compute_task_score() * 0.5

        reward = (
            mastery_gain * target_multiplier
            + eng_reward
            + prereq_bonus
            + zpd_bonus
            + correct_reward
            + terminal_bonus
            - fatigue_penalty
            - hint_penalty
            - overdrill_penalty
        )

        return float(np.clip(reward, 0.0, 1.0))

    def _check_done(self) -> bool:
        if self._step_count >= self.task_cfg["max_steps"]:
            return True
        # Early termination: engagement collapse
        if self._sim.state.engagement < 0.15:
            return True
        return False

    def _compute_task_score(self) -> float:
        """Final task score [0,1] used by grader."""
        fn = self.task_cfg["score_fn"]
        target    = self.task_cfg["target_concepts"]
        threshold = self.task_cfg["mastery_threshold"]
        mastery   = self._sim.state.mastery

        if fn == "single_concept":
            return float(np.clip(mastery[target[0]] / threshold, 0.0, 1.0))

        elif fn == "multi_concept":
            scores = [
                np.clip(mastery[c] / threshold, 0.0, 1.0) for c in target
            ]
            mastery_score = float(np.mean(scores))
            # Engagement multiplier
            eng_mult = np.clip(self._sim.state.engagement / 0.5, 0.5, 1.0)
            return float(mastery_score * eng_mult)

        elif fn == "exam_prep":
            scores = [
                np.clip(mastery[c] / threshold, 0.0, 1.0) for c in target
            ]
            mastery_score = float(np.mean(scores))
            # Efficiency: bonus for finishing under budget
            steps_used = self._step_count / self.task_cfg["max_steps"]
            efficiency = 1.0 - 0.2 * steps_used
            # Fatigue penalty
            fatigue_penalty = self._sim.state.fatigue * 0.15
            return float(np.clip(mastery_score * efficiency - fatigue_penalty, 0.0, 1.0))

        return 0.0

    # ── Utility ──────────────────────────────────────────

    @property
    def action_space(self) -> Dict:
        return {
            "type": "discrete_composite",
            "concept": {"type": "categorical", "values": CONCEPTS},
            "difficulty": {"type": "categorical", "values": DIFFICULTY_LEVELS},
            "hint_given": {"type": "boolean"},
        }

    @property
    def observation_space(self) -> Dict:
        n = len(CONCEPTS)
        return {
            "type": "dict",
            "concept_success_rates":    {"type": "float", "shape": (n,), "range": [0, 1]},
            "concept_attempt_counts":   {"type": "int",   "shape": (n,)},
            "concept_streaks":          {"type": "int",   "shape": (n,)},
            "prerequisite_readiness":   {"type": "float", "shape": (n,), "range": [0, 1]},
            "engagement":               {"type": "float", "shape": (1,), "range": [0, 1]},
            "fatigue":                  {"type": "float", "shape": (1,), "range": [0, 1]},
            "step_count":               {"type": "int",   "shape": (1,)},
        }

    def render(self, mode: str = "text") -> str:
        s = self._sim.state
        lines = [
            f"\n{'='*55}",
            f"  AdaptiveTutorEnv | Task: {self.task_id}",
            f"  Step: {self._step_count}/{self.task_cfg['max_steps']} | "
            f"Reward: {self._cum_reward:.3f}",
            f"  Engagement: {s.engagement:.2f} | Fatigue: {s.fatigue:.2f}",
            f"{'─'*55}",
            f"  {'Concept':<25} {'Mastery':>8} {'Attempts':>9} {'Streak':>7}",
            f"{'─'*55}",
        ]
        for c in CONCEPTS:
            total_att = sum(s.attempts[c].values())
            m = s.mastery[c]
            bar = "█" * int(m * 10) + "░" * (10 - int(m * 10))
            lines.append(
                f"  {c:<25} {bar} {m:.2f}  {total_att:>4}     {s.streak[c]:>3}"
            )
        lines.append(f"{'='*55}")
        return "\n".join(lines)