"""
NVIDIA Isaac Lab Style GPU-Native Matrix Multiplication

Isaac Lab keeps EVERYTHING on GPU:
- Observations stay in VRAM
- Actions stay in VRAM  
- Rewards computed on GPU
- No CPU-GPU transfers during training!

This is the ultimate performance pattern for RL.
"""

import torch
import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IsaacLabConfig:
    """
    Configuration for Isaac Lab style environment.
    
    Key settings:
    - num_envs: Number of parallel environments (all on GPU)
    - matrix_size: Size of matrices
    - device: Always "cuda" for Isaac Lab
    """
    num_envs: int = 1024
    matrix_size: int = 256
    device: str = "cuda"
    max_episode_length: int = 100
    
    # Isaac Lab specific
    return_observations_on_reset: bool = True
    return_rewards_on_reset: bool = False


class DirectRLMatMulEnv:
    """
    Isaac Lab DirectRL-style environment.
    
    Key differences from Gymnasium:
    1. ALL tensors stay on GPU (no .cpu() ever!)
    2. Vectorized by default (num_envs environments)
    3. Uses PyTorch tensors, not NumPy arrays
    4. Returns dict of tensors for flexibility
    
    This mirrors IsaacEnv/DirectRLEnv from Isaac Lab.
    """
    
    def __init__(self, cfg: IsaacLabConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.matrix_size = cfg.matrix_size
        
        # Episode tracking (on GPU!)
        self.episode_length = torch.zeros(
            cfg.num_envs, device=self.device, dtype=torch.int32
        )
        self.reset_buf = torch.zeros(
            cfg.num_envs, device=self.device, dtype=torch.bool
        )
        
        # State tensors (on GPU, never leave!)
        self.state = torch.zeros(
            cfg.num_envs, cfg.matrix_size, cfg.matrix_size,
            device=self.device, dtype=torch.float32
        )
        
        # Pre-allocate buffers for zero-allocation stepping
        self.rewards = torch.zeros(cfg.num_envs, device=self.device)
        self.dones = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.bool)
        
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environments.
        
        Isaac Lab style: can reset specific envs by ID (for auto-reset).
        Everything stays on GPU.
        """
        if env_ids is None:
            # Reset all environments
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(env_ids)
        
        # Reset state matrices (random initialization on GPU)
        self.state[env_ids] = torch.randn(
            num_reset, self.matrix_size, self.matrix_size,
            device=self.device, dtype=torch.float32
        )
        
        # Reset episode counters
        self.episode_length[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Return observation dict (Isaac Lab style)
        return {
            "obs": self.state.clone(),  # Clone to avoid aliasing
            "privileged_obs": None,      # For asymmetric actor-critic
        }
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments in parallel.
        
        Isaac Lab style:
        - Actions are (num_envs, matrix_size, matrix_size) tensor ON GPU
        - Returns tensors ON GPU
        - Auto-resets done environments
        
        Returns: (obs_dict, rewards, dones, info)
        """
        # Ensure actions are on the correct device
        if actions.device != self.device:
            actions = actions.to(self.device)
        
        # Matrix multiplication for ALL envs in parallel (batched)
        # This is a single CUDA kernel for all environments!
        self.state = torch.bmm(self.state, actions)
        
        # Compute rewards ON GPU (no CPU transfer!)
        # Reward = negative norm (want stable matrices)
        state_norms = torch.norm(self.state.view(self.num_envs, -1), dim=1)
        self.rewards = -state_norms / 1000.0  # Scale reward
        
        # Update episode length
        self.episode_length += 1
        
        # Check termination ON GPU
        diverged = state_norms > 1e6
        truncated = self.episode_length >= self.cfg.max_episode_length
        self.dones = diverged | truncated
        
        # Auto-reset done environments (Isaac Lab feature)
        reset_env_ids = self.dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)
        
        # Build observation dict
        obs_dict = {
            "obs": self.state.clone(),
            "privileged_obs": None,
        }
        
        info = {
            "episode_length": self.episode_length.clone(),
            "state_norms": state_norms,
        }
        
        return obs_dict, self.rewards, self.dones, info
    
    def close(self):
        """Cleanup (minimal for GPU tensors)."""
        pass


def benchmark_isaac_lab_style(
    num_envs_list=[64, 256, 1024, 4096],
    matrix_size=128,
    iterations=100
):
    """
    Benchmark Isaac Lab style environment.
    
    Key metric: Steps per second across all environments.
    Isaac Lab can often achieve millions of steps/second!
    """
    print("=" * 60)
    print("NVIDIA Isaac Lab Style GPU-Native Environment")
    print("=" * 60)
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    print()
    
    results = []
    
    for num_envs in num_envs_list:
        cfg = IsaacLabConfig(
            num_envs=num_envs,
            matrix_size=matrix_size,
            device="cuda"
        )
        
        env = DirectRLMatMulEnv(cfg)
        obs_dict = env.reset()
        
        # Create random actions (on GPU!)
        actions = torch.randn(
            num_envs, matrix_size, matrix_size,
            device="cuda", dtype=torch.float32
        ) * 0.01  # Small scale to avoid divergence
        
        # Warmup
        for _ in range(10):
            obs_dict, rewards, dones, info = env.step(actions)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            obs_dict, rewards, dones, info = env.step(actions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Metrics
        total_steps = num_envs * iterations
        steps_per_second = total_steps / elapsed
        time_per_step = elapsed / iterations * 1000  # ms
        
        # TFLOPS for matrix operations
        flops_per_step = num_envs * 2 * matrix_size ** 3
        total_flops = flops_per_step * iterations
        tflops = total_flops / elapsed / 1e12
        
        print(f"Num Envs: {num_envs:5d}")
        print(f"  Steps/second: {steps_per_second:,.0f}")
        print(f"  Time/step:    {time_per_step:.3f} ms")
        print(f"  TFLOPS:       {tflops:.2f}")
        print(f"  Mean reward:  {rewards.mean().item():.4f}")
        print()
        
        results.append({
            'num_envs': num_envs,
            'steps_per_second': steps_per_second,
            'time_per_step': time_per_step,
            'tflops': tflops
        })
        
        env.close()
    
    return results


def compare_gpu_native_vs_standard():
    """
    Compare Isaac Lab (GPU-native) vs Gymnasium (CPU-GPU transfers).
    
    This demonstrates why Isaac Lab is so much faster.
    """
    print("\n" + "=" * 60)
    print("GPU-Native vs Standard Comparison")
    print("=" * 60)
    
    num_envs = 256
    matrix_size = 128
    iterations = 100
    
    # === Isaac Lab Style (GPU-native) ===
    cfg = IsaacLabConfig(num_envs=num_envs, matrix_size=matrix_size)
    env_gpu = DirectRLMatMulEnv(cfg)
    env_gpu.reset()
    
    actions_gpu = torch.randn(
        num_envs, matrix_size, matrix_size, device="cuda"
    ) * 0.01
    
    # Warmup
    for _ in range(10):
        env_gpu.step(actions_gpu)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        obs, rew, done, info = env_gpu.step(actions_gpu)
    torch.cuda.synchronize()
    gpu_native_time = time.perf_counter() - start
    
    # === Simulated Standard Style (with CPU transfers) ===
    # This simulates what happens with standard Gym wrappers
    
    state = torch.randn(num_envs, matrix_size, matrix_size, device="cuda")
    
    # Warmup
    for _ in range(10):
        state = torch.bmm(state, actions_gpu)
        _ = state.cpu().numpy()  # Simulate Gym's observation return
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        state = torch.bmm(state, actions_gpu)
        obs_np = state.cpu().numpy()  # This is the slow part!
    torch.cuda.synchronize()
    standard_time = time.perf_counter() - start
    
    # Results
    print(f"GPU-Native (Isaac Lab): {gpu_native_time*1000:.1f} ms")
    print(f"Standard (with .cpu()): {standard_time*1000:.1f} ms")
    print(f"Speedup: {standard_time/gpu_native_time:.1f}x")
    
    env_gpu.close()


if __name__ == "__main__":
    benchmark_isaac_lab_style()
    compare_gpu_native_vs_standard()
