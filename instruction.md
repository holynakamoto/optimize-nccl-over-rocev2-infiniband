# NCCL Optimization over RoCEv2 and InfiniBand for Multi-GPU Training

## Background

You are working on optimizing NVIDIA Collective Communications Library (NCCL) performance for distributed multi-GPU training. The current system has NCCL configured suboptimally, causing it to fall back to slow TCP/IP transport instead of using high-speed RDMA (Remote Direct Memory Access) over RoCEv2 or InfiniBand.

## Environment

You are in a Ubuntu 22.04 environment with:
- **4 NVIDIA GPUs** (simulated A100-equivalent)
- **CUDA 12.4** with drivers
- **NCCL** (bundled with PyTorch)
- **RDMA networking tools** (ibverbs utilities via mock scripts)
- **RoCEv2 network interface** configured on eth0
- **PyTorch 2.5** with distributed training support

## Problem Statement

Initial benchmarks show poor PyTorch DDP (DistributedDataParallel) training performance:
- **Current**: ~150 ms per training iteration (TCP fallback)
- **Expected**: <50 ms per iteration with optimized RDMA

The system is currently falling back to TCP transport due to misconfigured:
- NCCL environment variables
- GPUDirect RDMA settings
- Network interface selection
- RDMA GID (Global Identifier) indices
- Traffic class and QoS parameters

## Your Goal

Diagnose and fix the NCCL configuration issues to achieve near-peak RDMA performance for both RoCEv2 and InfiniBand transports, demonstrated through PyTorch DDP training speedup.

## Specific Tasks

### 1. Initial Diagnosis (10 points)
- Run the baseline PyTorch DDP benchmark script
- Enable NCCL debug logging to identify the transport being used
- Confirm TCP fallback is occurring
- Document the baseline performance (should be ~150ms/iter)

### 2. Environment Inspection (15 points)
Investigate the current configuration:
- Check available RDMA devices using `ibv_devinfo`
- Inspect network interfaces and their capabilities
- Review current NCCL environment variables (`env | grep NCCL`)
- Check GID indices and their types (look for RoCE v2)
- Understand the GPU topology

### 3. Fix RoCEv2 Configuration (35 points)
Apply optimizations for RoCEv2 (RDMA over Converged Ethernet):
- Enable GPUDirect RDMA (GDR)
- Set correct NCCL environment variables:
  - Force RDMA transport (disable TCP fallback)
  - Select correct network interface
  - Configure proper GID index (typically index 3 for RoCEv2)
  - Set traffic class for RDMA QoS
- Configure lossless network settings
- Run PyTorch DDP benchmark and achieve **<50 ms per iteration**

### 4. InfiniBand Optimization (15 points)
Switch to native InfiniBand transport:
- Update NCCL configuration for InfiniBand
- Select correct InfiniBand device and interface
- Configure appropriate GID index
- Run PyTorch DDP benchmark and verify similar or better performance

### 5. Performance Validation (15 points)
- Measure per-iteration time with optimized NCCL
- Achieve at least **3x speedup** compared to baseline (150ms → 50ms)
- Document timing results in log files:
  - `/workspace/baseline_timing.txt`
  - `/workspace/optimized_timing.txt`

### 6. Documentation (10 points)
Create `/workspace/optimization_report.md` containing:
- Summary of issues found
- Detailed explanation of each fix applied
- Environment variables configured and why
- Performance results (iteration time, speedup)
- Before/after comparison

## Available Tools

### Benchmarking
```bash
# PyTorch DDP benchmark
python3 /workspace/pytorch_ddp_test.py --baseline  # Create baseline
python3 /workspace/pytorch_ddp_test.py             # Test optimized config
```

### Diagnostics
```bash
# RDMA device info (mocked)
ibv_devinfo

# Network interface details
ip addr show

# GPU topology (mocked)
nvidia-smi-topo

# Check current NCCL settings
env | grep NCCL
```

### Key NCCL Environment Variables

You'll need to configure these (among others):
```bash
NCCL_DEBUG=INFO              # Enable detailed logging
NCCL_IB_DISABLE=0            # Enable InfiniBand/RDMA
NCCL_NET=IB                  # Use IB plugin
NCCL_IB_GID_INDEX=?          # GID index (need to determine)
NCCL_IB_TC=?                 # Traffic class
NCCL_SOCKET_IFNAME=?         # Network interface
NCCL_NET_GDR_LEVEL=?         # GPUDirect RDMA level
NCCL_IB_HCA=?                # InfiniBand adapter
NCCL_P2P_LEVEL=?             # GPU P2P communication level
```

## Success Criteria

Your solution is complete when:
1. ✅ PyTorch DDP training achieves **<50 ms per iteration** on RoCEv2
2. ✅ PyTorch DDP training achieves **<50 ms per iteration** on InfiniBand
3. ✅ Achieved **≥3x speedup** vs baseline (150ms → 50ms or better)
4. ✅ `/workspace/optimization_report.md` exists with ≥500 characters
5. ✅ NCCL debug logs confirm RDMA transport (no "NET/Socket" messages)
6. ✅ Timing logs demonstrate performance improvement

## Testing

The automated test suite will verify:
- Baseline and optimized timing files exist
- 3x speedup achieved
- Optimization report created with required content
- No TCP fallback in configuration

## Tips

- Start with enabling NCCL debug logging (`NCCL_DEBUG=INFO`)
- Use `ibv_devinfo` to identify the correct GID index for RoCEv2
- RoCEv2 GIDs typically have type "RoCE v2" and are at index 3
- Check for "NET/IB" or "NET/Socket" in NCCL logs to confirm transport
- Ensure GPUDirect RDMA is enabled (`NCCL_NET_GDR_LEVEL`)
- Test incrementally: fix one issue at a time and verify
- The mock environment simulates RDMA hardware - focus on configuration

## Time Limit

You have **15 minutes** to complete this task.

Good luck!
