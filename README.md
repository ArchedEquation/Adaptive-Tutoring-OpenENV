---
title: AdaptiveTutor-v1
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🎓 AdaptiveTutor-v1 — OpenEnv Adaptive Student Tutoring Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-green.svg)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 26 passed](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](tests/)
[![PPO](https://img.shields.io/badge/RL-PPO%20trained-orange.svg)](baseline/rl_agent.py)

A **real-world, fully simulatable** OpenEnv environment where an AI agent acts as an
adaptive tutor for a student learning 10 high-school mathematics concepts.

The student's knowledge evolves using **Bayesian Knowledge Tracing (BKT)** — the same
model used in production by Khan Academy, Carnegie Learning, and Duolingo. The agent
must infer hidden mastery from observable responses and select optimal questions.

---

## 🧠 Environment Overview

```
Agent ──(concept, difficulty)──► Environment
                                     │
                          StudentSimulator (BKT)
                                     │
         ◄──(observation, reward, done)──────────
```

### The Challenge
- The student's **true mastery is hidden** — the agent observes only success rates, streaks, and engagement
- The agent must balance **exploration** (trying new concepts) vs **exploitation** (drilling weak areas)
- **Zone of Proximal Development**: questions too easy bore students; too hard discourages them
- **Prerequisite graph**: calculus requires functions; functions require algebra
- **Fatigue** accumulates — sessions must be managed efficiently

---

## 📐 Action Space

```python
TutorAction(
    concept:    str,   # one of 10 math concepts
    difficulty: str,   # "easy" | "medium" | "hard"
    hint_given: bool,  # give a hint (costs -0.05 reward)
)
```

**Concepts (10):** `algebra_basics`, `linear_equations`, `quadratic_equations`,
`functions`, `trigonometry`, `probability`, `statistics`, `calculus_intro`,
`geometry`, `number_theory`

---

## 👁️ Observation Space

```python
TutorObservation(
    concept_success_rates:    Dict[str, float],  # per-concept success rate [0,1]
    concept_attempt_counts:   Dict[str, int],    # total attempts per concept
    concept_streaks:          Dict[str, int],    # current correct streak
    prerequisite_readiness:   Dict[str, float],  # avg mastery of prerequisites [0,1]
    engagement:               float,             # student engagement [0,1]
    fatigue:                  float,             # accumulated fatigue [0,1]
    step_count:               int,               # steps this episode
    last_correct:             bool,              # was last answer correct?
    last_concept:             str,
    last_difficulty:          str,
)
```

---

## 🏆 Tasks

| Task | Difficulty | Steps | Goal | Score |
|------|-----------|-------|------|-------|
| `single_concept_mastery` | 🟢 Easy | 20 | Bring `algebra_basics` to ≥0.75 mastery | 0.0 – 1.0 |
| `multi_concept_curriculum` | 🟡 Medium | 50 | Master 5 concepts + keep engagement ≥0.5 | 0.0 – 1.0 |
| `exam_prep_sprint` | 🔴 Hard | 80 | Maximise all 10 concepts, manage fatigue + prerequisites | 0.0 – 1.0 |

---

## 💰 Reward Function

Dense reward at every step, composite of:

| Component | Weight | Description |
|-----------|--------|-------------|
| Mastery gain | ×3.0 | Change in true mastery (unmastered target concepts only) |
| Mastery gain | ×0.5 | Diminished return once concept is above threshold |
| Terminal bonus | +0.5× | Task score paid out at final step (long-horizon credit) |
| Correct answer | +0.10 | Base reward when student answers correctly |
| Engagement | +0.08 | Reward for maintaining student engagement |
| ZPD alignment | +0.08 | Zone of Proximal Development bonus |
| Prereq alignment | +0.05 | Prerequisites adequately mastered |
| Overdrill penalty | −0.08 | Wasting steps on already-mastered concepts |
| Fatigue penalty | −0.04 | Proportional to accumulated fatigue |
| Hint penalty | −0.05 | Per-step penalty for giving hints |

All rewards clipped to **[0.0, 1.0]**.

---

## 🚀 Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-tutor-env
cd adaptive-tutor-env
pip install -r requirements.txt
```

### Run one episode (Python)

```python
from env.environment import AdaptiveTutorEnv, TutorAction

env = AdaptiveTutorEnv(task_id="single_concept_mastery", seed=42)
obs = env.reset()

done = False
while not done:
    action = TutorAction(concept="algebra_basics", difficulty="medium")
    result = env.step(action)
    obs  = result.observation
    done = result.done
    print(f"reward={result.reward:.3f} | engagement={obs.engagement:.2f}")

print(env.render())
```

### Run via HTTP API

```bash
python app.py   # starts on port 7860
```

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "multi_concept_curriculum", "seed": 42}'

# Step  (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_ID", "concept": "algebra_basics", "difficulty": "medium"}'

# State
curl "http://localhost:7860/state?session_id=YOUR_ID"
```

### Run baseline evaluation

```bash
python baseline/baseline_inference.py --episodes 10 --seed 42
```

### Train + evaluate PPO agent

```bash
# Train on all 3 tasks
python baseline/rl_agent.py --train --task single_concept_mastery
python baseline/rl_agent.py --train --task multi_concept_curriculum
python baseline/rl_agent.py --train --task exam_prep_sprint

# Compare PPO vs all baselines
python baseline/rl_agent.py --eval --task all --episodes 10 --seed 42
```

### Run tests

```bash
pytest tests/ -v   # 26 tests, all passing
```

---

## 📊 Results (seed=42, 10 episodes)

PPO trained with Stable-Baselines3 **beats all hand-coded baselines** across every task:

| Agent | 🟢 Easy | 🟡 Medium | 🔴 Hard |
|-------|---------|-----------|---------|
| RandomAgent | 0.770 ±0.233 | 0.817 ±0.079 | 0.590 ±0.065 |
| HeuristicAgent | 0.930 ±0.163 | 0.979 ±0.034 | 0.670 ±0.026 |
| GreedyMasteryAgent | 0.930 ±0.163 | 0.895 ±0.061 | 0.686 ±0.021 |
| **PPO (RL)** | **0.932 ±0.175** | **1.000 ±0.002** | **0.701 ±0.012** |

- Medium task: PPO achieves **perfect 1.000** — learned an optimal curriculum strategy
- Hard task: PPO scores **0.701**, beating the best heuristic (0.686) with **3× lower variance**
- Hard task ceiling ~0.70 with these baselines — room for stronger RL algorithms

---

## 🔬 Student Model: Bayesian Knowledge Tracing

Each concept has 5 BKT parameters:

```
P(mastered_t+1 | mastered_t) = 1 - p_forget
P(mastered_t+1 | ~mastered_t) = p_learn
P(correct | mastered)  = 1 - p_slip
P(correct | ~mastered) = p_guess
```

Posterior update after each response:
```
P(mastered | correct) = P(mastered) * (1 - p_slip) / P(correct)
```

The agent **never observes mastery directly** — it must infer it from success rates,
streaks, and prerequisite readiness. This is what makes the environment genuinely hard.

---

## 📁 Project Structure

```
adaptive-tutor-env/
├── env/
│   ├── __init__.py
│   ├── student.py          # BKT student simulator (10 concepts, 5 params each)
│   ├── environment.py      # OpenEnv API: reset() / step() / state()
│   └── gym_wrapper.py      # Gymnasium wrapper for SB3/PPO training
├── graders/
│   └── grader.py           # EasyGrader, MediumGrader, HardGrader
├── baseline/
│   ├── agents.py           # RandomAgent, HeuristicAgent, GreedyMasteryAgent
│   ├── baseline_inference.py  # reproducible heuristic evaluation
│   └── rl_agent.py         # PPO train + eval + comparison script
├── tests/
│   └── test_env.py         # 26 unit + integration tests
├── openenv.yaml            # full OpenEnv spec
├── app.py                  # FastAPI server for HF Spaces
├── Dockerfile              # production container
├── requirements.txt
└── README.md
```

---

## 🐳 Docker

```bash
docker build -t adaptive-tutor-env .
docker run -p 7860:7860 adaptive-tutor-env
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + env info |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state?session_id=` | Episode state snapshot |
| GET | `/render?session_id=` | Text render of current state |
| GET | `/action_space` | Action space description |
| GET | `/observation_space` | Observation space description |
| GET | `/tasks` | List all tasks |
| DELETE | `/session/{id}` | Delete session |
| GET | `/docs` | Interactive Swagger UI |

---

## 🧩 Writing Your Own Agent

```python
from env.environment import AdaptiveTutorEnv
from env.student import CONCEPTS

def my_agent(obs: dict) -> dict:
    """obs is TutorObservation.model_dump()"""
    rates = obs["concept_success_rates"]
    # Pick concept with lowest success rate
    concept = min(CONCEPTS, key=lambda c: rates[c])
    # Adaptive difficulty
    sr = rates[concept]
    difficulty = "easy" if sr < 0.4 else "medium" if sr < 0.7 else "hard"
    return {"concept": concept, "difficulty": difficulty, "hint_given": False}

# Grade your agent
from graders.grader import run_all_graders
results = run_all_graders(my_agent, n_episodes=10)
```

---

## 📜 License

MIT
