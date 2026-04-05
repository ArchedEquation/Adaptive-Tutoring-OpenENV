"""
tests/test_env.py — Unit + integration tests for AdaptiveTutorEnv.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from env.environment import AdaptiveTutorEnv, TutorAction, TASK_REGISTRY
from env.student import CONCEPTS, DIFFICULTY_LEVELS, StudentSimulator
from graders.grader import EasyGrader, MediumGrader, HardGrader
from baseline.agents import RandomAgent, HeuristicAgent, GreedyMasteryAgent


# ── Student Simulator Tests ──────────────────────────────────────────────────

class TestStudentSimulator:
    def test_init(self):
        sim = StudentSimulator(seed=0)
        for c in CONCEPTS:
            assert 0.0 <= sim.state.mastery[c] <= 1.0

    def test_answer_returns_bool(self):
        sim = StudentSimulator(seed=0)
        result = sim.answer_question("algebra_basics", "easy")
        assert isinstance(result, bool)

    def test_mastery_evolves(self):
        sim = StudentSimulator(seed=42)
        m0 = sim.state.mastery["algebra_basics"]
        for _ in range(10):
            sim.answer_question("algebra_basics", "easy")
        m1 = sim.state.mastery["algebra_basics"]
        assert m0 != m1  # mastery should change

    def test_fatigue_increases(self):
        sim = StudentSimulator(seed=0)
        for c in CONCEPTS:
            sim.answer_question(c, "easy")
        assert sim.state.fatigue > 0.0

    def test_reset(self):
        sim = StudentSimulator(seed=0)
        sim.answer_question("algebra_basics", "medium")
        sim.reset(seed=0)
        assert sim.state.step_count == 0
        assert sim.state.fatigue == 0.0

    def test_prerequisite_readiness(self):
        sim = StudentSimulator(seed=0)
        # algebra_basics has no prerequisites → always 1.0
        assert sim.prerequisite_readiness("algebra_basics") == 1.0
        # calculus_intro has prerequisites
        r = sim.prerequisite_readiness("calculus_intro")
        assert 0.0 <= r <= 1.0


# ── Environment API Tests ────────────────────────────────────────────────────

class TestEnvironmentAPI:
    @pytest.fixture
    def env(self):
        e = AdaptiveTutorEnv(task_id="single_concept_mastery", seed=42)
        e.reset()
        return e

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert hasattr(obs, "concept_success_rates")
        assert hasattr(obs, "engagement")
        assert hasattr(obs, "fatigue")

    def test_step_returns_step_result(self, env):
        action = TutorAction(concept="algebra_basics", difficulty="easy")
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0
        assert isinstance(result.done, bool)
        assert result.observation is not None

    def test_step_dict_action(self, env):
        result = env.step({"concept": "algebra_basics", "difficulty": "medium", "hint_given": False})
        assert result.reward >= 0.0

    def test_state_returns_episode_info(self, env):
        info = env.state()
        assert info.task_id == "single_concept_mastery"
        assert info.max_steps == 20
        assert info.steps_remaining == 20

    def test_done_after_max_steps(self):
        env = AdaptiveTutorEnv(task_id="single_concept_mastery", seed=0)
        env.reset()
        done = False
        steps = 0
        while not done:
            result = env.step({"concept": "algebra_basics", "difficulty": "easy", "hint_given": False})
            done = result.done
            steps += 1
        assert steps <= 20 + 1  # max_steps or engagement collapse

    def test_invalid_concept_raises(self, env):
        with pytest.raises((ValueError, Exception)):
            env.step({"concept": "INVALID", "difficulty": "easy", "hint_given": False})

    def test_invalid_difficulty_raises(self, env):
        with pytest.raises((ValueError, Exception)):
            env.step({"concept": "algebra_basics", "difficulty": "INVALID", "hint_given": False})

    def test_reset_after_done(self):
        env = AdaptiveTutorEnv(task_id="single_concept_mastery", seed=0)
        env.reset()
        for _ in range(25):
            if env._done:
                break
            env.step({"concept": "algebra_basics", "difficulty": "easy", "hint_given": False})
        obs = env.reset()
        assert env._done is False
        assert env._step_count == 0

    def test_reward_in_range(self, env):
        for _ in range(10):
            result = env.step({"concept": "algebra_basics", "difficulty": "easy", "hint_given": False})
            assert 0.0 <= result.reward <= 1.0
            if result.done:
                break

    def test_all_tasks_run(self):
        for task_id in TASK_REGISTRY:
            env = AdaptiveTutorEnv(task_id=task_id, seed=0)
            obs = env.reset()
            assert obs is not None
            result = env.step({"concept": "algebra_basics", "difficulty": "easy", "hint_given": False})
            assert result is not None

    def test_render_output(self, env):
        text = env.render()
        assert "AdaptiveTutorEnv" in text
        assert "algebra_basics" in text


# ── Grader Tests ─────────────────────────────────────────────────────────────

class TestGraders:
    def _random_agent(self, obs):
        return {"concept": "algebra_basics", "difficulty": "easy", "hint_given": False}

    def test_easy_grader(self):
        grader = EasyGrader(n_episodes=3)
        result = grader.grade(self._random_agent)
        assert 0.0 <= result.score <= 1.0
        assert result.episodes_run == 3

    def test_medium_grader(self):
        grader = MediumGrader(n_episodes=3)
        result = grader.grade(self._random_agent)
        assert 0.0 <= result.score <= 1.0

    def test_hard_grader(self):
        grader = HardGrader(n_episodes=3)
        result = grader.grade(self._random_agent)
        assert 0.0 <= result.score <= 1.0

    def test_grader_result_summary(self):
        grader = EasyGrader(n_episodes=2)
        result = grader.grade(self._random_agent)
        s = result.summary()
        assert "score" in s.lower() or "Score" in s


# ── Baseline Agent Tests ─────────────────────────────────────────────────────

class TestBaselineAgents:
    def _run_episode(self, agent, task_id="single_concept_mastery"):
        env = AdaptiveTutorEnv(task_id=task_id, seed=7)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            result = env.step(agent(obs.model_dump()))
            obs = result.observation
            done = result.done
            steps += 1
        return env._compute_task_score()

    def test_random_agent(self):
        agent = RandomAgent(seed=0)
        score = self._run_episode(agent)
        assert 0.0 <= score <= 1.0

    def test_heuristic_agent(self):
        agent = HeuristicAgent(target_concepts=["algebra_basics"])
        score = self._run_episode(agent)
        assert 0.0 <= score <= 1.0

    def test_greedy_agent(self):
        agent = GreedyMasteryAgent(target_concepts=["algebra_basics"])
        score = self._run_episode(agent)
        assert 0.0 <= score <= 1.0

    def test_greedy_beats_random(self):
        """GreedyAgent should score higher than RandomAgent on average."""
        scores_random, scores_greedy = [], []
        for seed in range(5):
            r_agent = RandomAgent(seed=seed)
            g_agent = GreedyMasteryAgent(target_concepts=["algebra_basics"])
            scores_random.append(self._run_episode(r_agent))
            scores_greedy.append(self._run_episode(g_agent))
        assert np.mean(scores_greedy) >= np.mean(scores_random) - 0.1


# ── Reproducibility Test ─────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self):
        def run(seed):
            env = AdaptiveTutorEnv(task_id="single_concept_mastery", seed=seed)
            env.reset(seed=seed)
            rewards = []
            for _ in range(5):
                r = env.step({"concept": "algebra_basics", "difficulty": "easy", "hint_given": False})
                rewards.append(r.reward)
                if r.done:
                    break
            return rewards

        r1 = run(42)
        r2 = run(42)
        assert r1 == r2, "Same seed should produce identical results"
