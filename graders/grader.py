"""
Graders — evaluate an agent's performance on each task.
Each grader runs N episodes and returns a score in [0.0, 1.0].
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from env.environment import AdaptiveTutorEnv, TutorAction, TASK_REGISTRY
from env.student import CONCEPTS, DIFFICULTY_LEVELS


# ─────────────────────────────────────────────
# Grader Result
# ─────────────────────────────────────────────

@dataclass
class GraderResult:
    task_id: str
    difficulty: str
    score: float                  # final score [0,1]
    mean_task_score: float
    std_task_score: float
    mean_cumulative_reward: float
    mean_steps: float
    mean_final_engagement: float
    mean_final_fatigue: float
    episodes_run: int
    elapsed_seconds: float
    per_episode_scores: List[float]

    def summary(self) -> str:
        bar_len = int(self.score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        return (
            f"\n╔══════════════════════════════════════════════╗\n"
            f"║  Task     : {self.task_id:<33}║\n"
            f"║  Difficulty: {self.difficulty:<32}║\n"
            f"╠══════════════════════════════════════════════╣\n"
            f"║  Score  [{bar}] {self.score:.3f}  ║\n"
            f"║  Mean Task Score   : {self.mean_task_score:.4f} ± {self.std_task_score:.4f}     ║\n"
            f"║  Mean Cum. Reward  : {self.mean_cumulative_reward:.4f}               ║\n"
            f"║  Mean Steps        : {self.mean_steps:.1f}                   ║\n"
            f"║  Mean Engagement   : {self.mean_final_engagement:.4f}               ║\n"
            f"║  Mean Fatigue      : {self.mean_final_fatigue:.4f}               ║\n"
            f"║  Episodes          : {self.episodes_run:<25}║\n"
            f"╚══════════════════════════════════════════════╝"
        )


# ─────────────────────────────────────────────
# Agent type alias
# ─────────────────────────────────────────────

AgentFn = Callable[[Dict], Dict]  # obs dict -> action dict


# ─────────────────────────────────────────────
# Base Grader
# ─────────────────────────────────────────────

class BaseGrader:
    def __init__(self, task_id: str, n_episodes: int = 10, seeds: Optional[List[int]] = None):
        self.task_id    = task_id
        self.n_episodes = n_episodes
        self.seeds      = seeds or list(range(n_episodes))
        self.task_cfg   = TASK_REGISTRY[task_id]

    def grade(self, agent_fn: AgentFn) -> GraderResult:
        t0 = time.time()
        task_scores, cum_rewards, steps_list = [], [], []
        eng_list, fat_list = [], []

        for ep, seed in enumerate(self.seeds[:self.n_episodes]):
            env = AdaptiveTutorEnv(task_id=self.task_id, seed=seed, eval_mode=True)
            obs = env.reset(seed=seed)
            done = False

            while not done:
                action_dict = agent_fn(obs.model_dump())
                result = env.step(action_dict)
                obs    = result.observation
                done   = result.done

            ep_state  = env.state()
            task_score = env._compute_task_score()
            task_scores.append(task_score)
            cum_rewards.append(ep_state.cumulative_reward)
            steps_list.append(env._step_count)
            eng_list.append(env._sim.state.engagement)
            fat_list.append(env._sim.state.fatigue)

        score = self._aggregate(task_scores)
        return GraderResult(
            task_id=self.task_id,
            difficulty=self.task_cfg["difficulty"],
            score=score,
            mean_task_score=float(np.mean(task_scores)),
            std_task_score=float(np.std(task_scores)),
            mean_cumulative_reward=float(np.mean(cum_rewards)),
            mean_steps=float(np.mean(steps_list)),
            mean_final_engagement=float(np.mean(eng_list)),
            mean_final_fatigue=float(np.mean(fat_list)),
            episodes_run=self.n_episodes,
            elapsed_seconds=time.time() - t0,
            per_episode_scores=task_scores,
        )

    def _aggregate(self, scores: List[float]) -> float:
        """Default: trimmed mean (drop best + worst)."""
        if len(scores) <= 2:
            return float(np.mean(scores))
        sorted_s = sorted(scores)
        trimmed  = sorted_s[1:-1]
        return float(np.clip(np.mean(trimmed), 0.0, 1.0))


# ─────────────────────────────────────────────
# Task-specific Graders
# ─────────────────────────────────────────────

class EasyGrader(BaseGrader):
    """
    Task: single_concept_mastery (Easy)
    Score: mastery of algebra_basics / threshold, clipped to [0,1].
    Full score (1.0) requires bringing mastery to ≥0.75 in ≤20 steps.
    """
    def __init__(self, n_episodes: int = 10):
        super().__init__("single_concept_mastery", n_episodes)

    def _aggregate(self, scores: List[float]) -> float:
        # Mean score across all episodes — easy task should be reliable
        return float(np.clip(np.mean(scores), 0.0, 1.0))


class MediumGrader(BaseGrader):
    """
    Task: multi_concept_curriculum (Medium)
    Score: weighted by both mastery AND engagement retention.
    Partial credit for each concept above 0.5 mastery.
    """
    def __init__(self, n_episodes: int = 10):
        super().__init__("multi_concept_curriculum", n_episodes)

    def grade(self, agent_fn: AgentFn) -> GraderResult:
        result = super().grade(agent_fn)
        # Additional penalty if agent frequently collapses engagement
        low_eng_penalty = max(0, 0.6 - result.mean_final_engagement) * 0.3
        result.score = float(np.clip(result.score - low_eng_penalty, 0.0, 1.0))
        return result


class HardGrader(BaseGrader):
    """
    Task: exam_prep_sprint (Hard)
    Score: composite of:
      - Average mastery across all 10 concepts
      - Step efficiency (fewer steps = better)
      - Fatigue management (fatigue < 0.8 required for full score)
      - Prerequisite order adherence bonus
    """
    def __init__(self, n_episodes: int = 10):
        super().__init__("exam_prep_sprint", n_episodes)

    def grade(self, agent_fn: AgentFn) -> GraderResult:
        t0 = time.time()
        task_scores, cum_rewards, steps_list = [], [], []
        eng_list, fat_list = [], []
        prereq_adherence_list = []

        for ep, seed in enumerate(self.seeds[:self.n_episodes]):
            env = AdaptiveTutorEnv(task_id=self.task_id, seed=seed, eval_mode=True)
            obs = env.reset(seed=seed)
            done = False
            prereq_violations = 0
            total_actions = 0

            while not done:
                action_dict = agent_fn(obs.model_dump())
                concept    = action_dict.get("concept", CONCEPTS[0])
                difficulty = action_dict.get("difficulty", "easy")

                # Track prerequisite adherence
                prereq_ready = env._sim.prerequisite_readiness(concept)
                if prereq_ready < 0.3:
                    prereq_violations += 1
                total_actions += 1

                result = env.step(action_dict)
                obs  = result.observation
                done = result.done

            task_score = env._compute_task_score()
            # Adherence bonus: fewer violations = higher bonus (up to 0.1)
            adherence = 1.0 - prereq_violations / max(total_actions, 1)
            prereq_adherence_list.append(adherence)

            task_scores.append(task_score)
            cum_rewards.append(env.state().cumulative_reward)
            steps_list.append(env._step_count)
            eng_list.append(env._sim.state.engagement)
            fat_list.append(env._sim.state.fatigue)

        base_score  = float(np.clip(np.mean(task_scores), 0.0, 1.0))
        prereq_bonus = float(np.mean(prereq_adherence_list)) * 0.1
        fat_penalty  = max(0, float(np.mean(fat_list)) - 0.8) * 0.2
        score = float(np.clip(base_score + prereq_bonus - fat_penalty, 0.0, 1.0))

        return GraderResult(
            task_id=self.task_id,
            difficulty=self.task_cfg["difficulty"],
            score=score,
            mean_task_score=float(np.mean(task_scores)),
            std_task_score=float(np.std(task_scores)),
            mean_cumulative_reward=float(np.mean(cum_rewards)),
            mean_steps=float(np.mean(steps_list)),
            mean_final_engagement=float(np.mean(eng_list)),
            mean_final_fatigue=float(np.mean(fat_list)),
            episodes_run=self.n_episodes,
            elapsed_seconds=time.time() - t0,
            per_episode_scores=task_scores,
        )


# ─────────────────────────────────────────────
# Grader Registry
# ─────────────────────────────────────────────

GRADER_REGISTRY: Dict[str, type] = {
    "single_concept_mastery":    EasyGrader,
    "multi_concept_curriculum":  MediumGrader,
    "exam_prep_sprint":          HardGrader,
}


def run_all_graders(agent_fn: AgentFn, n_episodes: int = 10) -> Dict[str, GraderResult]:
    results = {}
    for task_id, GraderClass in GRADER_REGISTRY.items():
        print(f"\n  ▶ Grading task: {task_id} ...")
        grader = GraderClass(n_episodes=n_episodes)
        results[task_id] = grader.grade(agent_fn)
        print(results[task_id].summary())
    return results
