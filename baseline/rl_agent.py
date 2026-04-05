"""
rl_agent.py — Train a PPO agent on AdaptiveTutorEnv using Stable-Baselines3.

Usage:
    # Train on all 3 tasks
    python baseline/rl_agent.py --train

    # Train on a specific task
    python baseline/rl_agent.py --train --task exam_prep_sprint

    # Evaluate saved model vs baselines
    python baseline/rl_agent.py --eval --task exam_prep_sprint

    # Full pipeline: train + eval + compare
    python baseline/rl_agent.py --train --eval --task exam_prep_sprint
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from env.gym_wrapper import AdaptiveTutorGymEnv, int_to_action
from env.environment import TASK_REGISTRY
from env.student import CONCEPTS
from graders.grader import GRADER_REGISTRY
from baseline.agents import RandomAgent, HeuristicAgent, GreedyMasteryAgent


# ── Config ────────────────────────────────────────────────────────────────────

TASK_TRAIN_CONFIG = {
    "single_concept_mastery": {
        "total_timesteps": 60_000,
        "n_envs": 4,
        "policy_kwargs": dict(net_arch=[64, 64]),
    },
    "multi_concept_curriculum": {
        "total_timesteps": 120_000,
        "n_envs": 4,
        "policy_kwargs": dict(net_arch=[128, 128]),
    },
    "exam_prep_sprint": {
        "total_timesteps": 250_000,
        "n_envs": 4,
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
    },
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Progress callback ─────────────────────────────────────────────────────────

class TrainingProgressCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log    = 0
        self._t0          = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            elapsed = time.time() - self._t0
            fps     = self.num_timesteps / max(elapsed, 1)
            if self.verbose:
                print(
                    f"  step={self.num_timesteps:>7,} | "
                    f"fps={fps:>5.0f} | "
                    f"elapsed={elapsed:>5.0f}s"
                )
            self._last_log = self.num_timesteps
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train_ppo(task_id: str, seed: int = 42) -> PPO:
    cfg = TASK_TRAIN_CONFIG[task_id]
    print(f"\n{'='*60}")
    print(f"  Training PPO on: {task_id}")
    print(f"  Timesteps : {cfg['total_timesteps']:,}")
    print(f"  Parallel envs: {cfg['n_envs']}")
    print(f"{'='*60}")

    # No VecNormalize -- obs already in [0,1], normalisation adds noise
    vec_env = make_vec_env(
        lambda: AdaptiveTutorGymEnv(task_id=task_id, seed=seed),
        n_envs=cfg["n_envs"],
        seed=seed,
    )

    eval_env = make_vec_env(
        lambda: AdaptiveTutorGymEnv(task_id=task_id, seed=seed + 999),
        n_envs=1,
        seed=seed + 999,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=max(10_000 // cfg["n_envs"], 1),
        n_eval_episodes=10,
        verbose=0,
    )
    progress_cb = TrainingProgressCallback(log_interval=50_000)

    # n_steps covers at least 2 full episodes per env for credit assignment
    max_steps  = TASK_REGISTRY[task_id]["max_steps"]
    n_steps    = max(512, max_steps * 4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=2e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,        # care deeply about future mastery gains
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,      # encourage exploration of all concepts
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=cfg["policy_kwargs"],
        verbose=0,
        seed=seed,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[eval_cb, progress_cb],
        progress_bar=False,
    )
    elapsed = time.time() - t0

    # Save model (no vecnorm file needed)
    save_path = os.path.join(MODEL_DIR, f"ppo_{task_id}")
    model.save(save_path)

    print(f"\n  Training done in {elapsed:.0f}s")
    print(f"  Model saved -> {save_path}.zip")
    return model



class PPOAgent:
    """Wraps a trained SB3 PPO model into an AgentFn callable."""
    name = "PPO (trained)"

    def __init__(self, task_id: str):
        model_path = os.path.join(MODEL_DIR, f"ppo_{task_id}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run --train first."
            )
        self._model     = PPO.load(model_path)
        self._task_id   = task_id
        self._max_steps = TASK_REGISTRY[task_id]["max_steps"]

    def __call__(self, obs_dict: dict) -> dict:
        from env.gym_wrapper import obs_to_vector
        vec = obs_to_vector(obs_dict, self._max_steps)
        action, _ = self._model.predict(vec, deterministic=True)
        concept, difficulty = int_to_action(int(action))
        return {"concept": concept, "difficulty": difficulty, "hint_given": False}



def evaluate_all(task_id: str, n_episodes: int = 10, seed: int = 42) -> dict:
    GraderClass = GRADER_REGISTRY[task_id]
    task_cfg    = TASK_REGISTRY[task_id]

    results = {}

    # Baseline agents
    baseline_agents = {
        "Random":    RandomAgent(seed=seed),
        "Heuristic": HeuristicAgent(target_concepts=task_cfg["target_concepts"]),
        "Greedy":    GreedyMasteryAgent(target_concepts=task_cfg["target_concepts"]),
    }

    for name, agent in baseline_agents.items():
        print(f"  Grading {name:12s} ...", end=" ", flush=True)
        grader = GraderClass(n_episodes=n_episodes)
        r = grader.grade(agent)
        results[name] = r.score
        print(f"score={r.score:.4f} ±{r.std_task_score:.4f}")

    # PPO agent
    try:
        ppo_agent = PPOAgent(task_id)
        print(f"  Grading PPO        ...", end=" ", flush=True)
        grader = GraderClass(n_episodes=n_episodes)
        r = grader.grade(ppo_agent)
        results["PPO"] = r.score
        print(f"score={r.score:.4f} ±{r.std_task_score:.4f}")
    except FileNotFoundError as e:
        print(f"  PPO model not found: {e}")

    return results


def print_comparison(task_id: str, results: dict):
    diff = TASK_REGISTRY[task_id]["difficulty"]
    icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[diff]
    print(f"\n  {'─'*50}")
    print(f"  {icon}  {task_id}  ({diff})")
    print(f"  {'─'*50}")
    best = max(results.values())
    for name, score in results.items():
        bar = "█" * int(score * 25)
        tag = " ← 🏆 best" if score == best else ""
        print(f"  {name:<14} [{bar:<25}] {score:.4f}{tag}")
    print(f"  {'─'*50}")
    if "PPO" in results:
        baseline_best = max(v for k, v in results.items() if k != "PPO")
        delta = results["PPO"] - baseline_best
        sign  = "+" if delta >= 0 else ""
        print(f"  PPO vs best baseline: {sign}{delta:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPO RL Agent for AdaptiveTutorEnv")
    parser.add_argument("--train",    action="store_true", help="Train PPO")
    parser.add_argument("--eval",     action="store_true", help="Evaluate + compare agents")
    parser.add_argument("--task",     type=str, default="all",
                        choices=["all"] + list(TASK_REGISTRY))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default=None)
    args = parser.parse_args()

    tasks = list(TASK_REGISTRY) if args.task == "all" else [args.task]

    all_results = {}

    for task_id in tasks:
        if args.train:
            train_ppo(task_id, seed=args.seed)

        if args.eval:
            print(f"\n  Evaluating task: {task_id}")
            r = evaluate_all(task_id, n_episodes=args.episodes, seed=args.seed)
            all_results[task_id] = r
            print_comparison(task_id, r)

    if args.output and all_results:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved → {args.output}")

    if not args.train and not args.eval:
        parser.print_help()


if __name__ == "__main__":
    main()