# NCCL Optimization Task for Terminal-Bench 2.0

This task challenges AI agents to diagnose and optimize NVIDIA NCCL (Collective Communications Library) performance for multi-GPU distributed training over RDMA networks (RoCEv2 and InfiniBand).

## Task Overview

**Difficulty**: Hard  
**Estimated Time**: 15 minutes  
**Domain**: HPC/AI Infrastructure, Distributed Systems, GPU Computing

Agents must identify and fix misconfigurations causing NCCL to fall back to slow TCP transport, then optimize for high-performance RDMA transport over both RoCEv2 and InfiniBand.

## Success Criteria

1. ✅ Achieve ≥180 GB/s bandwidth on RoCEv2
2. ✅ Achieve ≥190 GB/s bandwidth on InfiniBand
3. ✅ Achieve ≥3x speedup in PyTorch DDP training
4. ✅ Create comprehensive optimization report (≥500 chars)
5. ✅ No TCP fallback in NCCL logs

## Directory Structure

```
optimize-nccl-over-RoCEv2-InfiniBand/
├── task.toml                      # Task configuration
├── instruction.md                 # Student-facing instructions
├── README.md                      # This file
│
├── environment/                   # Docker environment setup
│   ├── Dockerfile                 # CUDA, NCCL, OpenMPI, PyTorch
│   ├── setup_mock_rdma.sh         # Mock RDMA devices
│   ├── mock_ibv_devinfo.sh        # Mock ibverbs command
│   ├── mock_nvidia_smi.sh         # Mock GPU topology
│   ├── baseline_benchmark.sh      # Initial (broken) benchmark
│   ├── pytorch_ddp_test.py        # PyTorch training test
│   └── optimization_report_template.md
│
├── solution/
│   └── solve.sh                   # Reference solution
│
└── tests/
    ├── test.sh                    # Test runner script
    └── test_outputs.py            # pytest test suite
```

## Key Learning Objectives

1. **NCCL Configuration**: Understanding NCCL environment variables and their impact
2. **RDMA Fundamentals**: GID indices, traffic classes, GPUDirect RDMA
3. **Network Optimization**: RoCEv2 vs InfiniBand, lossless Ethernet (PFC/ECN)
4. **Distributed Training**: How NCCL affects PyTorch DDP performance
5. **Systems Debugging**: Using logs and diagnostic tools to identify issues

## Technical Challenges

### Initial Misconfigurations
- `NCCL_IB_DISABLE=1` forces TCP fallback
- `NCCL_SOCKET_IFNAME=lo` uses wrong interface
- Missing `NCCL_IB_GID_INDEX` configuration
- GPUDirect RDMA not enabled

### Required Knowledge
- NCCL environment variable system
- RDMA GID types and selection
- RoCEv2 vs native InfiniBand differences
- GPUDirect RDMA benefits
- NCCL debugging and logging

## Environment Details

### Simulated Hardware
- 4x NVIDIA A100-equivalent GPUs
- 2x Mellanox ConnectX-6 adapters
  - mlx5_0: RoCEv2 on eth0
  - mlx5_1: InfiniBand on ib0

### Software Stack
- Ubuntu 22.04
- CUDA 12.4
- NCCL 2.22+
- OpenMPI 5.x
- PyTorch 2.5.1 with CUDA support
- RDMA tools (ibverbs, perftest)

## Testing Approach

The test suite (`tests/test_outputs.py`) validates:

1. **File Artifacts**: Presence of timing logs and report
2. **Performance Metrics**: Bandwidth and speedup thresholds
3. **Configuration Quality**: NCCL environment variables
4. **Transport Verification**: No TCP fallback in logs

## Mock Environment

Since running actual multi-GPU RDMA benchmarks requires expensive hardware, this task uses:

1. **Mock RDMA devices**: Scripts simulate `ibv_devinfo` output
2. **Simulated benchmarks**: Reference results demonstrate expected performance
3. **Configuration focus**: Emphasis on knowing correct NCCL settings
4. **Validation via artifacts**: Tests check for correct configuration files

This approach tests **knowledge and reasoning** about NCCL optimization while remaining hardware-independent.

## Difficulty Justification

**Expected Pass Rate**: 40-60% for frontier AI agents

**Why Hard?**
1. **Niche Knowledge**: RDMA/NCCL expertise rare in training data
2. **Multi-Variable Problem**: Many interconnected configuration issues
3. **Debugging Required**: Must interpret logs and environment state
4. **Domain-Specific**: HPC/AI infrastructure is specialized
5. **Easy to Miss Details**: Wrong GID index, traffic class, or interface breaks everything

**Why Solvable by Experts?**
- All information is in public NVIDIA documentation
- Standard practices in ML infrastructure engineering
- Logical debugging process with clear error messages
- Step-by-step diagnosis possible

## Real-World Relevance

This task mirrors actual problems faced by:
- ML infrastructure engineers at hyperscalers (Meta, Google, Microsoft)
- HPC cluster administrators
- AI training platform developers
- GPU cluster optimization specialists

Skills tested are directly applicable to:
- Optimizing large-scale model training (LLaMA, GPT-4, etc.)
- Building ML infrastructure platforms
- Debugging distributed training performance issues
- Configuring GPU clusters for AI workloads

## Extensions & Variations

Potential task variations for different difficulty levels:

**Easier Version**:
- Pre-identify which variables are wrong
- Provide hints about GID index selection
- Smaller set of issues to fix

**Harder Version**:
- Add NUMA affinity problems
- Include PCIe topology issues
- Multiple network fabrics with complex routing
- Performance regression debugging

**Related Tasks**:
- Optimize NCCL for NVSwitch topologies
- Debug NCCL hangs in large-scale training
- Configure multi-node NCCL with InfiniBand
- Optimize all-to-all communication patterns

## References

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [RoCEv2 Configuration Best Practices](https://enterprise-support.nvidia.com/s/article/understanding-rocev2-congestion-management)
- [GPUDirect RDMA Guide](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Meta's OCP Experience with GPU Networks](https://engineering.fb.com/2021/03/31/data-center-engineering/networking-ocp/)

## Contributing

To improve this task:
1. Test with actual NCCL benchmarks if GPU hardware available
2. Add more realistic network topology scenarios
3. Include additional diagnostic tools
4. Expand test coverage for edge cases
5. Add progressive hints system for educational use

## License

This task is part of Terminal-Bench 2.0 and follows the benchmark's licensing terms.
