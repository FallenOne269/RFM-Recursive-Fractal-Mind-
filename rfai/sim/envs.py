from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class TaskHierarchyEnv:
    rng_seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        self.step_count = 0
        self.progress = 0
        self.num_subtasks = 3

    def reset(self):
        self.step_count = 0
        self.progress = 0
        return {"progress": self.progress}

    def step(self):
        delay = self.rng.integers(0, 2)
        self.step_count += 1
        if delay == 0 and self.progress < self.num_subtasks:
            self.progress += 1
        done = self.progress >= self.num_subtasks
        return {"progress": self.progress, "done": done}


@dataclass
class ObstacleAvoidanceEnv:
    grid_size: int = 5
    rng_seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        self.pos = (0, 0)
        self.steps = 0
        self.obstacles = {(2, 2)}

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return {"pos": self.pos}

    def step(self):
        move = self.rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
        new_pos = (max(0, min(self.grid_size - 1, self.pos[0] + move[0])), max(0, min(self.grid_size - 1, self.pos[1] + move[1])))
        self.obstacles = {( (o[0] + 1) % self.grid_size, (o[1] + 1) % self.grid_size) for o in self.obstacles}
        hit = new_pos in self.obstacles
        self.pos = new_pos
        self.steps += 1
        return {"pos": self.pos, "hit": hit}


@dataclass
class ResourceAllocationEnv:
    num_slots: int = 3
    rng_seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        self.alloc = np.zeros(self.num_slots)

    def reset(self):
        self.alloc = np.zeros(self.num_slots)
        return {"alloc": self.alloc.copy()}

    def step(self):
        add = self.rng.random(self.num_slots)
        self.alloc += add
        efficiency = float(np.exp(-np.std(self.alloc)))
        return {"alloc": self.alloc.copy(), "efficiency": efficiency}

