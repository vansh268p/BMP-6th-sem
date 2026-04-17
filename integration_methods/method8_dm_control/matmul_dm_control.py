"""
DeepMind Control Suite Style Matrix Multiplication

DM Control uses a richer API than Gym:
- TimeStep namedtuple instead of plain tuples
- Explicit step_type (FIRST, MID, LAST)
- Cleaner separation of concerns
- Physics-based timesteps

This demonstrates the DM Control patterns.
"""

import numpy as np
import torch
from enum import IntEnum
from typing import NamedTuple, Optional
from dataclasses import dataclass
import time


class StepType(IntEnum):
    """
    Step types following DM Control conventions.
    
    FIRST: Initial step after reset
    MID: Normal step during episode
    LAST: Terminal step
    """
    FIRST = 0
    MID = 1
    LAST = 2


class TimeStep(NamedTuple):
    """
    DM Control style TimeStep.
    
    More structured than Gym's (obs, reward, done, info) tuple.
    Makes it explicit what each field means.
    """
    step_type: StepType
    reward: Optional[float]
    discount: Optional[float]
    observation: np.ndarray
    
    def first(self) -> bool:
        return self.step_type == StepType.FIRST
    
    def mid(self) -> bool:
        return self.step_type == StepType.MID
    
    def last(self) -> bool:
        return self.step_type == StepType.LAST


@dataclass
class ActionSpec:
    """Specification for valid actions."""
    shape: tuple
    dtype: np.dtype
    minimum: float
    maximum: float
    name: str = "action"


@dataclass
class ObservationSpec:
    """Specification for observations."""
    shape: tuple
    dtype: np.dtype
    name: str = "observation"


class MatMulTask:
    """
    DM Control style task specification.
    
    Separates the "task" (what reward to compute) from 
    the "physics" (how the simulation works).
    """
    
    def __init__(self, target_norm=1.0):
        self.target_norm = target_norm
    
    def get_reward(self, state: torch.Tensor) -> float:
        """Compute reward based on state."""
        current_norm = torch.norm(state).item()
        # Reward for being close to target norm
        return -abs(current_norm - self.target_norm)
    
    def get_termination(self, state: torch.Tensor) -> bool:
        """Check if episode should terminate."""
        return torch.norm(state).item() > 1e6  # Diverged


class MatMulPhysics:
    """
    DM Control style physics simulation.
    
    Handles the actual computation (matrix multiplication).
    Separate from task/reward logic.
    """
    
    def __init__(self, matrix_size: int, device: str = "cuda"):
        self.matrix_size = matrix_size
        self.device = device
        self.state = None
        self._timestep = 0
    
    def reset(self) -> torch.Tensor:
        """Reset physics state."""
        self.state = torch.randn(
            self.matrix_size, self.matrix_size,
            dtype=torch.float32, device=self.device
        )
        self._timestep = 0
        return self.state
    
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Apply action (matrix multiply)."""
        self.state = torch.mm(self.state, action)
        self._timestep += 1
        return self.state
    
    def timestep(self) -> int:
        return self._timestep


class MatMulEnvironment:
    """
    DM Control style environment.
    
    Combines Physics (simulation) with Task (rewards).
    Returns TimeStep objects instead of tuples.
    """
    
    def __init__(self, matrix_size: int = 1024, device: str = "cuda"):
        self.physics = MatMulPhysics(matrix_size, device)
        self.task = MatMulTask()
        self.matrix_size = matrix_size
        self._max_steps = 100
        
    def action_spec(self) -> ActionSpec:
        """Return specification for valid actions."""
        return ActionSpec(
            shape=(self.matrix_size, self.matrix_size),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name="transformation_matrix"
        )
    
    def observation_spec(self) -> ObservationSpec:
        """Return specification for observations."""
        return ObservationSpec(
            shape=(self.matrix_size, self.matrix_size),
            dtype=np.float32,
            name="state_matrix"
        )
    
    def reset(self) -> TimeStep:
        """Reset environment and return initial TimeStep."""
        state = self.physics.reset()
        observation = state.cpu().numpy()
        
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,  # No reward on first step
            discount=None,
            observation=observation
        )
    
    def step(self, action: np.ndarray) -> TimeStep:
        """
        Take a step and return TimeStep.
        
        This is the key difference from Gym:
        - Returns structured TimeStep, not tuple
        - step_type explicitly indicates episode state
        - discount is explicit (for proper RL)
        """
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).to(self.physics.device)
        else:
            action_tensor = action.to(self.physics.device)
        
        # Physics step
        state = self.physics.step(action_tensor)
        
        # Task evaluation
        reward = self.task.get_reward(state)
        terminated = self.task.get_termination(state)
        truncated = self.physics.timestep() >= self._max_steps
        
        # Determine step type
        if terminated or truncated:
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0
        
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=state.cpu().numpy()
        )


def benchmark_dm_control_style(sizes=[512, 1024, 2048], iterations=10):
    """Benchmark DM Control style environment."""
    print("=" * 60)
    print("DeepMind Control Suite Style Matrix Multiplication")
    print("=" * 60)
    
    results = []
    
    for size in sizes:
        env = MatMulEnvironment(matrix_size=size, device="cuda")
        
        # Print specs (DM Control style)
        print(f"\nSize {size}×{size}:")
        print(f"  Action spec: {env.action_spec()}")
        print(f"  Observation spec: {env.observation_spec()}")
        
        # Reset and get initial TimeStep
        timestep = env.reset()
        assert timestep.first(), "First timestep should be FIRST type"
        
        # Create action
        action = np.random.randn(size, size).astype(np.float32) * 0.01
        
        # Warmup
        for _ in range(3):
            timestep = env.step(action)
            if timestep.last():
                timestep = env.reset()
        torch.cuda.synchronize()
        
        # Reset for benchmark
        timestep = env.reset()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            timestep = env.step(action)
            if timestep.last():
                timestep = env.reset()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
        
        flops = 2 * size * size * size
        tflops = flops / elapsed / 1e12
        
        print(f"  Time: {elapsed*1000:.3f} ms, {tflops:.2f} TFLOPS")
        print(f"  Last reward: {timestep.reward:.4f}")
        
        results.append({
            'size': size,
            'time': elapsed,
            'tflops': tflops
        })
    
    return results


if __name__ == "__main__":
    benchmark_dm_control_style()
