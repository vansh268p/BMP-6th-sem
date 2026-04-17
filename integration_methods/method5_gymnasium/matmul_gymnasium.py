"""
Gymnasium-Style Matrix Multiplication Wrapper

Gymnasium (successor to OpenAI Gym) uses a standardized API for environments.
This demonstrates wrapping C++ matrix multiplication in that pattern.

Key Concepts:
- step() / reset() pattern
- Observation/Action spaces
- Standardized interface for RL libraries
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class MatMulEnv(gym.Env):
    """
    A Gymnasium environment that wraps matrix multiplication.
    
    This demonstrates how to wrap a C++ computation in Gym's API.
    In RL, this could represent a physics simulation where:
    - observation = state matrix
    - action = transformation matrix
    - step() = apply transformation (matrix multiply)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, matrix_size=1024, device="cuda"):
        super().__init__()
        
        self.matrix_size = matrix_size
        self.device = device
        
        # Define observation and action spaces
        # In Gym, these define the valid inputs/outputs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(matrix_size, matrix_size),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(matrix_size, matrix_size),
            dtype=np.float32
        )
        
        # Internal state
        self.state = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize state matrix
        self.state = torch.randn(
            self.matrix_size, self.matrix_size,
            dtype=torch.float32, device=self.device
        )
        
        observation = self.state.cpu().numpy()
        info = {"matrix_size": self.matrix_size}
        
        return observation, info
    
    def step(self, action):
        """
        Apply action (matrix) to state via matrix multiplication.
        
        In RL terms:
        - action: the transformation matrix
        - new_state = state @ action
        """
        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).to(self.device)
        else:
            action_tensor = action.to(self.device)
        
        # Matrix multiplication (the actual computation)
        self.state = torch.mm(self.state, action_tensor)
        
        # Gymnasium API returns: observation, reward, terminated, truncated, info
        observation = self.state.cpu().numpy()
        reward = -torch.norm(self.state).item()  # Example reward
        terminated = False
        truncated = False
        info = {"state_norm": torch.norm(self.state).item()}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional)."""
        if self.state is not None:
            print(f"State matrix shape: {self.state.shape}")
            print(f"State norm: {torch.norm(self.state).item():.4f}")


def benchmark_gymnasium(sizes=[512, 1024, 2048], iterations=10):
    """Benchmark Gymnasium-style matrix multiplication."""
    import time
    
    print("=" * 60)
    print("Gymnasium-Style Matrix Multiplication Benchmark")
    print("=" * 60)
    
    results = []
    
    for size in sizes:
        env = MatMulEnv(matrix_size=size, device="cuda")
        obs, info = env.reset()
        
        # Create action (transformation matrix)
        action = np.random.randn(size, size).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            env.step(action)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            obs, reward, term, trunc, info = env.step(action)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
        
        flops = 2 * size * size * size
        tflops = flops / elapsed / 1e12
        
        print(f"Size {size}×{size}: {elapsed*1000:.3f} ms, {tflops:.2f} TFLOPS")
        results.append((size, elapsed, tflops))
    
    return results


if __name__ == "__main__":
    benchmark_gymnasium()
