"""
inference.py — AdaptiveTutor-v1 OpenEnv Inference Script

Uses an LLM (via OpenAI-compatible API) as the tutoring agent.
The LLM observes student state and decides which concept + difficulty to present.

Environment variables:
    API_BASE_URL   LLM endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key
"""

import os
import json
import sys
from typing import List, Optional

from openai import OpenAI

# ── Env vars ──────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

TASKS     = ["single_concept_mastery", "multi_concept_curriculum", "exam_prep_sprint"]
BENCHMARK = "adaptive-tutor-v1"

CONCEPTS     = [
    "algebra_basics", "linear_equations", "quadratic_equations", "functions",
    "trigonometry", "probability", "statistics", "calculus_intro",
    "geometry", "number_theory",
]
DIFFICULTIES = ["easy", "medium", "hard"]

TASK_META = {
    "single_concept_mastery": {
        "target_concepts": ["algebra_basics"],
        "goal": "Bring algebra_basics to mastery (>=0.75) in 20 steps.",
        "max_steps": 20,
    },
    "multi_concept_curriculum": {
        "target_concepts": ["algebra_basics", "linear_equations", "functions", "probability", "geometry"],
        "goal": "Bring 5 concepts to mastery (>=0.70) in 50 steps while keeping engagement above 0.5.",
        "max_steps": 50,
    },
    "exam_prep_sprint": {
        "target_concepts": CONCEPTS,
        "goal": "Maximise mastery across all 10 concepts in 80 steps. Respect prerequisites. Manage fatigue.",
        "max_steps": 80,
    },
}

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(task_id: str) -> str:
    meta    = TASK_META[task_id]
    targets = ", ".join(meta["target_concepts"])
    return f"""You are an expert AI tutor agent playing task: {task_id}

TASK GOAL: {meta["goal"]}
TARGET CONCEPTS (focus here): {targets}
MAX STEPS: {meta["max_steps"]}

You observe the student's state and pick the next concept + difficulty.

STRATEGY:
- Focus ONLY on target concepts unless fatigue is high
- Match difficulty to success rate: easy if sr<0.4, medium if sr<0.7, hard if sr>=0.7
- Avoid concepts already mastered (sr>0.85) — switch to next weakest target
- If fatigue>0.7, always pick easy difficulty
- Respect prerequisites: only teach a concept if its prereq_readiness >= 0.4

Respond with ONLY a JSON object, no explanation, no markdown:
{{"concept": "<concept_name>", "difficulty": "<easy|medium|hard>", "hint_given": false}}

Valid concepts: algebra_basics, linear_equations, quadratic_equations, functions,
trigonometry, probability, statistics, calculus_intro, geometry, number_theory"""


def build_user_prompt(obs: dict, step: int, task_id: str) -> str:
    meta     = TASK_META[task_id]
    targets  = set(meta["target_concepts"])
    rates    = obs["concept_success_rates"]
    streaks  = obs["concept_streaks"]
    prereqs  = obs["prerequisite_readiness"]
    eng      = obs["engagement"]
    fatigue  = obs["fatigue"]
    last_ok  = obs["last_correct"]
    last_c   = obs["last_concept"]
    last_d   = obs["last_difficulty"]
    max_s    = meta["max_steps"]

    rows = []
    for c in CONCEPTS:
        sr     = rates.get(c, 0.0)
        st     = streaks.get(c, 0)
        pr     = prereqs.get(c, 1.0)
        is_tgt = "★" if c in targets else " "
        rows.append(f"  {is_tgt} {c:<28} sr={sr:.2f}  streak={st}  prereq={pr:.2f}")
    concept_table = "\n".join(rows)

    return f"""Step {step}/{max_s} | Engagement: {eng:.2f} | Fatigue: {fatigue:.2f}
Last: {last_c} ({last_d}) → {'CORRECT ✓' if last_ok else 'WRONG ✗'}

Concept state (★ = target):
{concept_table}

Pick the next concept and difficulty. JSON only:"""


# ── LLM action ────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, step: int, task_id: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt(task_id)},
                {"role": "user",   "content": build_user_prompt(obs, step, task_id)},
            ],
            temperature=0.1,
            max_tokens=60,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        action = json.loads(text)
        assert action["concept"]    in CONCEPTS,     f"Bad concept: {action['concept']}"
        assert action["difficulty"] in DIFFICULTIES, f"Bad difficulty: {action['difficulty']}"
        return {"concept": action["concept"], "difficulty": action["difficulty"], "hint_given": False}
    except Exception as e:
        # Fallback: heuristic — pick weakest unmastered target concept
        meta    = TASK_META[task_id]
        targets = meta["target_concepts"]
        rates   = obs["concept_success_rates"]
        prereqs = obs["prerequisite_readiness"]
        fatigue = obs["fatigue"]
        eligible = [
            c for c in targets
            if prereqs.get(c, 1.0) >= 0.35 and rates.get(c, 0.0) < 0.85
        ] or targets
        concept = min(eligible, key=lambda c: rates.get(c, 0.0))
        sr      = rates.get(concept, 0.0)
        diff    = "easy" if fatigue > 0.7 or sr < 0.4 else "medium" if sr < 0.7 else "hard"
        return {"concept": concept, "difficulty": diff, "hint_given": False}


# ── Single task runner ────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> dict:
    from env.environment import AdaptiveTutorEnv

    max_steps = TASK_META[task_id]["max_steps"]
    env       = AdaptiveTutorEnv(task_id=task_id, seed=42, eval_mode=True)

    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    error_msg   = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = env.reset(seed=42)
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action     = get_llm_action(client, obs.model_dump(), step, task_id)
            action_str = f"concept={action['concept']},difficulty={action['difficulty']}"

            try:
                result = env.step(action)
                obs    = result.observation
                reward = result.reward
                done   = result.done
                error_msg = None
            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

        score   = env._compute_task_score()
        success = score >= 0.5

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_id, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("[ERROR] No API key. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
        sys.exit(1)

    client    = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_env  = os.getenv("ADAPTIVE_TUTOR_TASK", "all")
    tasks     = TASKS if task_env == "all" else [task_env]

    all_results = []
    for task_id in tasks:
        result = run_task(task_id, client)
        all_results.append(result)

    print("\n[SUMMARY]", flush=True)
    for r in all_results:
        print(
            f"  task={r['task']} score={r['score']:.3f} "
            f"success={str(r['success']).lower()} steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()