# AGENTS.md
RFM / Recursive Fractal Mind — Agent System Specification

> Status: draft v0.1 • Scope: define agent roles, message protocol, lifecycle, tooling, safety, and ops for the RFM/FRM stack (RZ-OS alignment).

---

## 1) Overview

**Goal.** Provide a composable, testable agent architecture that wraps the FRM core with role-specialized behaviors, shared memory, and tool use. Agents communicate via a minimal JSON protocol; the Orchestrator coordinates planning and delegation. Ethical DNA guards every loop.

**Core ideas**
- **Agents are policies over loops.** Each agent is a policy configured with goals, constraints, tools, and memory access. The loop is perceive → recall → plan → act → reflect → update → recurse.
- **Small, explicit contracts.** Agents talk in one JSON schema. Tools follow a single callable contract. Memory has tiered stores with clear retention rules.
- **Safety by default.** Ethical DNA checks run pre- and post-action. Red-team tests live in `tests/`.

---

## 2) Architecture at a glance
"[User/API] ─▶ [Orchestrator]
│
├─▶ [Planner] ─▶ plan graph (DAG of steps)
│ │
│ ├─▶ [Tool Broker] ─▶ tools (web, code, files, etc.)
│ └─▶ [Worker Agents] (Researcher, Synthesizer, Critic, etc.)
│
├─▶ [Ethical DNA] (policy checks; RZ-OS guardrails)
├─▶ [Memory Service] (short/long/episodic/symbolic)
└─▶ [Telemetry] (traces, metrics, artifacts)"


