"""
app.py — FastAPI server for AdaptiveTutorEnv on Hugging Face Spaces.
Exposes the full OpenEnv API over HTTP with session management.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import AdaptiveTutorEnv, TutorAction, TASK_REGISTRY
from env.student import CONCEPTS, DIFFICULTY_LEVELS


# ── Session store (in-memory, fine for demo) ─────────────────────────────────

SESSIONS: Dict[str, AdaptiveTutorEnv] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    SESSIONS.clear()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AdaptiveTutorEnv",
    description=(
        "OpenEnv-compliant Adaptive Student Tutoring Environment. "
        "Simulates a student learning 10 math concepts via Bayesian Knowledge Tracing. "
        "API: reset() / step() / state()"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "single_concept_mastery"
    seed: int = 42
    eval_mode: bool = False


class StepRequest(BaseModel):
    session_id: str
    concept: str
    difficulty: str
    hint_given: bool = False


class SessionResponse(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check + env info")
def root():
    return {
        "env_id": AdaptiveTutorEnv.ENV_ID,
        "version": AdaptiveTutorEnv.VERSION,
        "tasks": list(TASK_REGISTRY.keys()),
        "concepts": CONCEPTS,
        "difficulties": DIFFICULTY_LEVELS,
        "api": ["/reset", "/step", "/state", "/render", "/docs"],
        "status": "ok",
    }


@app.post("/reset", summary="Reset environment, start new episode")
def reset(req: ResetRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task_id: {req.task_id}")

    session_id = str(uuid.uuid4())
    env = AdaptiveTutorEnv(
        task_id=req.task_id,
        seed=req.seed,
        eval_mode=req.eval_mode,
    )
    obs = env.reset(seed=req.seed)
    SESSIONS[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "task": {
            "id": req.task_id,
            "difficulty": TASK_REGISTRY[req.task_id]["difficulty"],
            "max_steps": TASK_REGISTRY[req.task_id]["max_steps"],
            "description": TASK_REGISTRY[req.task_id]["description"],
        },
    }


@app.post("/step", summary="Take one tutoring action")
def step(req: StepRequest):
    env = _get_session(req.session_id)
    if env._done:
        raise HTTPException(400, "Episode is done. Call /reset to start a new one.")

    try:
        action = TutorAction(
            concept=req.concept,
            difficulty=req.difficulty,
            hint_given=req.hint_given,
        )
    except Exception as e:
        raise HTTPException(422, str(e))

    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state", summary="Get current episode state")
def state(session_id: str = Query(...)):
    env = _get_session(session_id)
    return env.state().model_dump()


@app.get("/render", summary="Render current state as text")
def render(session_id: str = Query(...)):
    env = _get_session(session_id)
    return {"render": env.render()}


@app.get("/action_space", summary="Describe the action space")
def action_space():
    env = AdaptiveTutorEnv()
    return env.action_space


@app.get("/observation_space", summary="Describe the observation space")
def observation_space():
    env = AdaptiveTutorEnv()
    return env.observation_space


@app.get("/tasks", summary="List all available tasks")
def tasks():
    return {
        task_id: {
            "difficulty":   cfg["difficulty"],
            "description":  cfg["description"],
            "max_steps":    cfg["max_steps"],
            "target_concepts": cfg["target_concepts"],
        }
        for task_id, cfg in TASK_REGISTRY.items()
    }


@app.delete("/session/{session_id}", summary="Delete a session")
def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    del SESSIONS[session_id]
    return {"deleted": session_id}


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> AdaptiveTutorEnv:
    if session_id not in SESSIONS:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /reset first.")
    return SESSIONS[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
