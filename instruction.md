# NCCL Optimization over RoCEv2 and InfiniBand for Multi-GPU Training

## Background

You are working on optimizing NVIDIA Collective Communications Library (NCCL) performance for distributed multi-GPU training. The current system has NCCL configured suboptimally, causing it to fall back to slow TCP/IP transport instead of using high-speed RDMA (Remote Direct Memory Access) over RoCEv2 or InfiniBand.

## Environment

You are in a Ubuntu 22.04 environment with:
- **4 NVIDIA GPUs** (simulated A100-equivalent)
- **CUDA 12.4** with drivers
- **NCCL 2.22+** (NVIDIA Collective Communications Library)
- **nccl-tests** suite (pre-built for benchmarking)
- **OpenMPI 5.x** for multi-process coordination
- **RDMA networking tools** (ibverbs, perftest)
- **RoCEv2 network interface** configured on eth0
- **PyTorch with distributed training support**

## Problem Statement

Initial benchmarks show poor NCCL AllReduce performance:
- **Current**: ~20-30 GB/s aggregated bandwidth
- **Expected**: 180+ GB/s on optimized RoCEv2, 190+ GB/s on InfiniBand

The system is currently falling back to TCP transport due to misconfigured:
- NCCL environment variables
- GPUDirect RDMA settings
- Network interface selection
- RDMA GID (Global Identifier) indices
- Traffic class and QoS parameters

## Your Goal

Diagnose and fix the NCCL configuration issues to achieve near-peak RDMA performance for both RoCEv2 and InfiniBand transports.

## Specific Tasks

### 1. Initial Diagnosis (5 points)
- Run the baseline benchmark using `nccl-tests`
- Enable NCCL debug logging to identify the transport being used
- Confirm TCP fallback is occurring
- Document the baseline performance

### 2. Environment Inspection (10 points)
Investigate the current configuration:
- Check available RDMA devices (`ibv_devinfo`, `ibv_devices`)
- Inspect network interfaces and their capabilities
- Review current NCCL environment variables
- Check GID indices and their types
- Verify GPU topology and P2P capabilities

### 3. Fix RoCEv2 Configuration (35 points)
Apply optimizations for RoCEv2 (RDMA over Converged Ethernet):
- Enable GPUDirect RDMA (GDR)
- Set correct NCCL environment variables:
  - Force RDMA transport (disable TCP fallback)
  - Select correct network interface
  - Configure proper GID index (typically index 3 for RoCEv2)
  - Set traffic class for RDMA QoS
- Configure lossless network:
  - Priority Flow Control (PFC) settings
  - ECN (Explicit Congestion Notification) thresholds
- Run benchmarks and achieve **>180 GB/s** aggregated bandwidth

### 4. InfiniBand Optimization (20 points)
Switch to native InfiniBand transport:
- Update NCCL configuration for InfiniBand
- Optimize for multi-rail topology
- Configure traffic class appropriately
- Run benchmarks and achieve **>190 GB/s** aggregated bandwidth

### 5. PyTorch Validation (20 points)
- Run the provided PyTorch DDP (DistributedDataParallel) training script
- Measure per-iteration time with optimized NCCL
- Achieve at least **3x speedup** compared to baseline
- Document timing results in separate log files

### 6. Documentation (10 points)
Create `optimization_report.md` containing:
- Summary of issues found
- Detailed explanation of each fix applied
- Environment variables configured and why
- Performance results (bandwidth, speedup)
- Before/after comparison

## Available Tools

### Benchmarking
```bash
# NCCL benchmarks (in /opt/nccl-tests/build/)
./all_reduce_perf -b 8 -e 8G -f 2 -g 1

# RDMA bandwidth test
ib_write_bw -d mlx5_0 -a -F --report_gbits

# Network performance
iperf3 -c <host> -P 4
```

### Diagnostics
```bash
# RDMA device info
ibv_devinfo
ibv_devices

# Network interface details
ip addr show
ethtool <interface>
ibstat

# GPU info
nvidia-smi topo -m
nvidia-smi

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
1. ✅ RoCEv2 benchmarks show **≥180 GB/s** aggregated bandwidth
2. ✅ InfiniBand benchmarks show **≥190 GB/s** aggregated bandwidth
3. ✅ PyTorch DDP training shows **≥3x speedup** vs baseline
4. ✅ `optimization_report.md` exists with ≥500 characters
5. ✅ NCCL debug logs confirm RDMA transport (no "NET/Socket" messages)
6. ✅ Timing logs demonstrate performance improvement

## Testing

Run the automated verification:
```bash
/tests/test.sh
```

This will execute all checks and generate a pass/fail report.

## Tips

- Start with enabling NCCL debug logging (`NCCL_DEBUG=INFO`)
- Use `ibv_devinfo` to identify the correct GID index for RoCEv2
- RoCEv2 GIDs typically have type "RoCE v2" and are at index 3
- Check for "NET/IB" or "NET/Socket" in NCCL logs to confirm transport
- Ensure GPUDirect RDMA is enabled at the kernel level
- Test incrementally: fix one issue at a time and verify

## Time Limit

You have **15 minutes** to complete this task.

Good luck!
