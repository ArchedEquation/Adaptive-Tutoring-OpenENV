"""
baseline_inference.py — Reproducible baseline evaluation script.

Usage:
    python baseline_inference.py                    # run all agents, all tasks
    python baseline_inference.py --agent greedy     # specific agent
    python baseline_inference.py --task single_concept_mastery
    python baseline_inference.py --episodes 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np

from env.environment import AdaptiveTutorEnv, TASK_REGISTRY
from env.student import CONCEPTS
from graders.grader import GRADER_REGISTRY, run_all_graders, GraderResult
from baseline.agents import get_agent, RandomAgent, HeuristicAgent, GreedyMasteryAgent


AGENT_CONFIGS = {
    "random":    {"cls": RandomAgent,       "kwargs": {"seed": 0}},
    "heuristic": {"cls": HeuristicAgent,    "kwargs": {}},
    "greedy":    {"cls": GreedyMasteryAgent,"kwargs": {}},
}

TASK_AGENT_MAP = {
    "single_concept_mastery":   {"heuristic": {"target_concepts": ["algebra_basics"]}},
    "multi_concept_curriculum": {
        "heuristic": {"target_concepts": [
            "algebra_basics", "linear_equations", "functions",
            "probability", "geometry"
        ]},
        "greedy": {"target_concepts": [
            "algebra_basics", "linear_equations", "functions",
            "probability", "geometry"
        ]},
    },
    "exam_prep_sprint": {
        "greedy": {"target_concepts": CONCEPTS},
    },
}


def run_single_episode(task_id: str, agent, seed: int = 0, render: bool = False) -> dict:
    """Run one episode and return metrics."""
    env  = AdaptiveTutorEnv(task_id=task_id, seed=seed, eval_mode=True)
    obs  = env.reset(seed=seed)
    done = False
    step = 0

    while not done:
        action = agent(obs.model_dump())
        result = env.step(action)
        obs    = result.observation
        done   = result.done
        step  += 1

        if render and step % 10 == 0:
            print(env.render())

    return {
        "task_score":  env._compute_task_score(),
        "cum_reward":  env.state().cumulative_reward,
        "steps":       step,
        "engagement":  env._sim.state.engagement,
        "fatigue":     env._sim.state.fatigue,
        "mastery":     env._sim.get_mastery_summary(),
    }


def evaluate_agent_on_task(
    task_id: str,
    agent_name: str,
    n_episodes: int = 10,
    base_seed: int = 42,
) -> dict:
    """Evaluate one agent on one task across n_episodes seeds."""
    task_kwargs = TASK_AGENT_MAP.get(task_id, {}).get(agent_name, {})
    agent_cls   = AGENT_CONFIGS[agent_name]["cls"]
    base_kwargs = AGENT_CONFIGS[agent_name]["kwargs"].copy()
    base_kwargs.update(task_kwargs)
    agent = agent_cls(**base_kwargs)

    scores, rewards, steps_list, eng_list, fat_list = [], [], [], [], []

    for i in range(n_episodes):
        seed = base_seed + i
        ep   = run_single_episode(task_id, agent, seed=seed)
        scores.append(ep["task_score"])
        rewards.append(ep["cum_reward"])
        steps_list.append(ep["steps"])
        eng_list.append(ep["engagement"])
        fat_list.append(ep["fatigue"])

    return {
        "agent":          agent_name,
        "task":           task_id,
        "difficulty":     TASK_REGISTRY[task_id]["difficulty"],
        "n_episodes":     n_episodes,
        "mean_score":     float(np.mean(scores)),
        "std_score":      float(np.std(scores)),
        "min_score":      float(np.min(scores)),
        "max_score":      float(np.max(scores)),
        "mean_reward":    float(np.mean(rewards)),
        "mean_steps":     float(np.mean(steps_list)),
        "mean_engagement":float(np.mean(eng_list)),
        "mean_fatigue":   float(np.mean(fat_list)),
        "per_episode_scores": scores,
    }


def print_results_table(all_results: list[dict]):
    print("\n" + "═" * 80)
    print("  BASELINE EVALUATION RESULTS")
    print("═" * 80)
    print(f"  {'Agent':<18} {'Task':<30} {'Diff':<8} {'Score':>7} {'±':>5} {'Steps':>6}")
    print("─" * 80)
    for r in all_results:
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(r["difficulty"], "")
        print(
            f"  {r['agent']:<18} {r['task']:<30} "
            f"{diff_icon} {r['difficulty']:<6} "
            f"{r['mean_score']:>6.3f}  "
            f"±{r['std_score']:.3f}  "
            f"{r['mean_steps']:>5.0f}"
        )
    print("═" * 80)


def main():
    parser = argparse.ArgumentParser(description="AdaptiveTutorEnv Baseline Evaluation")
    parser.add_argument("--agent",    type=str, default="all",
                        choices=["all"] + list(AGENT_CONFIGS), help="Agent to evaluate")
    parser.add_argument("--task",     type=str, default="all",
                        choices=["all"] + list(TASK_REGISTRY), help="Task to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per eval")
    parser.add_argument("--seed",     type=int, default=42,  help="Base random seed")
    parser.add_argument("--render",   action="store_true",   help="Render one episode")
    parser.add_argument("--output",   type=str, default=None,help="Save results to JSON")
    args = parser.parse_args()

    agents = list(AGENT_CONFIGS) if args.agent == "all" else [args.agent]
    tasks  = list(TASK_REGISTRY) if args.task  == "all" else [args.task]

    print(f"\n  AdaptiveTutorEnv v1.0.0 — Baseline Evaluation")
    print(f"  Agents: {agents}")
    print(f"  Tasks : {tasks}")
    print(f"  Episodes per eval: {args.episodes} | Seed: {args.seed}\n")

    if args.render:
        print("  ── Demo Episode (GreedyAgent, exam_prep_sprint) ──")
        agent = GreedyMasteryAgent(target_concepts=CONCEPTS)
        run_single_episode("exam_prep_sprint", agent, seed=args.seed, render=True)

    all_results = []
    t0 = time.time()

    for task_id in tasks:
        for agent_name in agents:
            print(f"  Evaluating {agent_name} on {task_id} ...", end=" ", flush=True)
            r = evaluate_agent_on_task(task_id, agent_name, args.episodes, args.seed)
            all_results.append(r)
            print(f"score={r['mean_score']:.3f} ±{r['std_score']:.3f}")

    print_results_table(all_results)
    print(f"\n  Total time: {time.time()-t0:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
