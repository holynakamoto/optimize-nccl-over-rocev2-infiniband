#!/usr/bin/env python3
"""
PyTorch DDP (DistributedDataParallel) test script for NCCL optimization task.
This simulates a multi-GPU training workload to measure NCCL performance.
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training environment."""
    # Get rank and world size from environment or use defaults
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Initialize process group with NCCL backend
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def create_model():
    """Create a simple model for testing."""
    # Large enough model to generate significant NCCL traffic
    model = nn.Sequential(
        nn.Linear(2048, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 2048),
    ).cuda()

    return model


def run_training_benchmark(num_iterations=50, warmup_iterations=10):
    """Run a training benchmark and measure iteration time."""

    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("PyTorch DDP Training Benchmark")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print(f"CUDA device: {torch.cuda.get_device_name(local_rank)}")
        print("")
        print("NCCL Environment Variables:")
        for key in sorted(os.environ.keys()):
            if "NCCL" in key:
                print(f"  {key}={os.environ[key]}")
        print("")

    # Create model and wrap with DDP
    model = create_model()
    model = DDP(model, device_ids=[local_rank])

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Loss function
    criterion = nn.MSELoss()

    # Dummy dataset
    batch_size = 128
    input_size = 2048

    if rank == 0:
        print("Running warmup iterations...")

    # Warmup iterations
    for i in range(warmup_iterations):
        data = torch.randn(batch_size, input_size).cuda()
        target = torch.randn(batch_size, 2048).cuda()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize before timing
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"Running {num_iterations} timed iterations...")

    # Timed iterations
    iteration_times = []

    for i in range(num_iterations):
        data = torch.randn(batch_size, input_size).cuda()
        target = torch.randn(batch_size, 2048).cuda()

        start_time = time.time()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Ensure all NCCL operations complete
        torch.cuda.synchronize()

        iteration_time = time.time() - start_time
        iteration_times.append(iteration_time)

        if rank == 0 and (i + 1) % 10 == 0:
            avg_time = sum(iteration_times[-10:]) / 10
            print(f"  Iterations {i - 8}-{i + 1}: avg {avg_time * 1000:.2f} ms/iter")

    # Calculate statistics
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    min_iteration_time = min(iteration_times)
    max_iteration_time = max(iteration_times)

    # Synchronize before gathering results
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("")
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Average iteration time: {avg_iteration_time * 1000:.2f} ms")
        print(f"Min iteration time:     {min_iteration_time * 1000:.2f} ms")
        print(f"Max iteration time:     {max_iteration_time * 1000:.2f} ms")
        print(f"Throughput:             {1.0 / avg_iteration_time:.2f} iter/s")
        print("=" * 60)

        # Save timing results
        with open("/workspace/optimized_timing.txt", "w") as f:
            f.write(f"{avg_iteration_time}\n")

        print("")
        print("Timing saved to /workspace/optimized_timing.txt")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return avg_iteration_time


def simulate_baseline():
    """Simulate baseline performance with TCP fallback."""
    # This represents the slow performance with TCP/IP transport
    # Typically 3-5x slower than optimized RDMA
    baseline_time = 0.150  # 150ms per iteration (simulated slow performance)

    print("=" * 60)
    print("Simulated Baseline Performance (TCP Fallback)")
    print("=" * 60)
    print(f"Average iteration time: {baseline_time * 1000:.2f} ms")
    print("=" * 60)

    with open("/workspace/baseline_timing.txt", "w") as f:
        f.write(f"{baseline_time}\n")

    print("Baseline timing saved to /workspace/baseline_timing.txt")

    return baseline_time


if __name__ == "__main__":
    # Check if we're creating baseline or running optimized benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        simulate_baseline()
    else:
        try:
            run_training_benchmark()
        except Exception as e:
            print(f"Error running benchmark: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)
