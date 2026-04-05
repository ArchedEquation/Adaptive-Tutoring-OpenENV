"""
gym_wrapper.py — Gymnasium-compatible wrapper around AdaptiveTutorEnv.
Converts the OpenEnv dict API into standard Box/Discrete spaces
so any SB3/RLlib agent can train on it directly.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from env.environment import AdaptiveTutorEnv, TutorAction, TASK_REGISTRY
from env.student import CONCEPTS, DIFFICULTY_LEVELS, PREREQUISITES


# ── Encoding helpers ──────────────────────────────────────────────────────────

N_CONCEPTS    = len(CONCEPTS)          # 10
N_DIFFICULTIES = len(DIFFICULTY_LEVELS) # 3
N_ACTIONS     = N_CONCEPTS * N_DIFFICULTIES  # 30  (no hint — keeps action space clean)

CONCEPT_IDX   = {c: i for i, c in enumerate(CONCEPTS)}
DIFF_IDX      = {d: i for i, d in enumerate(DIFFICULTY_LEVELS)}

def action_to_int(concept: str, difficulty: str) -> int:
    return CONCEPT_IDX[concept] * N_DIFFICULTIES + DIFF_IDX[difficulty]

def int_to_action(action_int: int) -> Tuple[str, str]:
    concept_i   = action_int // N_DIFFICULTIES
    difficulty_i = action_int %  N_DIFFICULTIES
    return CONCEPTS[concept_i], DIFFICULTY_LEVELS[difficulty_i]


# ── Observation builder ───────────────────────────────────────────────────────

OBS_DIM = N_CONCEPTS * 4 + 4
# Per concept: success_rate, norm_attempts, streak_norm, prereq_readiness
# Global:      engagement, fatigue, steps_remaining_norm, last_correct

def obs_to_vector(obs_dict: dict, max_steps: int) -> np.ndarray:
    vec = []
    for c in CONCEPTS:
        vec.append(obs_dict["concept_success_rates"].get(c, 0.0))
        vec.append(min(obs_dict["concept_attempt_counts"].get(c, 0) / 20.0, 1.0))
        vec.append(min(obs_dict["concept_streaks"].get(c, 0) / 5.0, 1.0))
        vec.append(obs_dict["prerequisite_readiness"].get(c, 1.0))
    vec.append(obs_dict["engagement"])
    vec.append(obs_dict["fatigue"])
    steps_done = obs_dict["step_count"]
    vec.append(1.0 - steps_done / max_steps)   # steps remaining fraction
    vec.append(1.0 if obs_dict["last_correct"] else 0.0)
    return np.array(vec, dtype=np.float32)


# ── Gymnasium Wrapper ─────────────────────────────────────────────────────────

class AdaptiveTutorGymEnv(gym.Env):
    """
    Gymnasium wrapper for AdaptiveTutorEnv.
    Observation: flat float32 vector of shape (OBS_DIM,)
    Action:      Discrete(30) — 10 concepts × 3 difficulties
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_id: str = "exam_prep_sprint",
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.task_id     = task_id
        self._base_seed  = seed
        self.render_mode = render_mode
        self._max_steps  = TASK_REGISTRY[task_id]["max_steps"]

        self._env = AdaptiveTutorEnv(task_id=task_id, seed=seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        _seed = seed if seed is not None else self._base_seed
        obs = self._env.reset(seed=_seed)
        return obs_to_vector(obs.model_dump(), self._max_steps), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        concept, difficulty = int_to_action(int(action))
        result = self._env.step({
            "concept":    concept,
            "difficulty": difficulty,
            "hint_given": False,
        })
        obs_vec  = obs_to_vector(result.observation.model_dump(), self._max_steps)
        reward   = float(result.reward)
        done     = result.done
        info     = result.info
        info["task_score"] = self._env._compute_task_score()
        return obs_vec, reward, done, False, info  # truncated=False

    def render(self):
        if self.render_mode in ("human", "ansi"):
            print(self._env.render())

    def close(self):
        pass