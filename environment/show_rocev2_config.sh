#!/bin/bash
# Helper script to show RoCEv2 network configuration
# This simulates ethtool and sysctl output for educational purposes

cat << 'EOF'
RoCEv2 Network Configuration
=============================

Interface: eth0 (mlx5_0)
------------------------
Link: UP
Speed: 200000Mb/s (200 Gbps)
Duplex: Full

Priority Flow Control (PFC):
  Priority 0: disabled
  Priority 1: disabled
  Priority 2: disabled
  Priority 3: disabled
  Priority 4: disabled
  Priority 5: enabled   <-- Traffic Class 5
  Priority 6: disabled
  Priority 7: disabled

Explicit Congestion Notification (ECN):
  ECN: enabled
  DSCP to Priority mapping:
    DSCP 26 -> Priority 5 (TC 5)

  ECN Thresholds for Priority 5:
    Minimum Threshold: 150 KB
    Maximum Threshold: 1500 KB
    Probability: 100%

RDMA Configuration:
  GID Index 3: RoCE v2 (Active)
  MTU: 4096
  Congestion Control: DCQCN (Data Center Quantized Congestion Notification)

QoS Configuration:
  Traffic Class 5 Settings:
    - PFC enabled (lossless)
    - ECN marking enabled
    - Priority level: High
    - Queue depth: 128 KB

Why TC 5 with PFC/ECN?
----------------------
1. PFC prevents packet drops by sending pause frames
2. ECN marks packets during congestion instead of dropping
3. Together they create lossless Ethernet for RDMA
4. TC 5 is standard in many datacenter RoCEv2 deployments
5. Ensures RDMA reliability over Ethernet fabric

NCCL Configuration Required:
----------------------------
export NCCL_IB_TC=5              # Use traffic class 5
export NCCL_IB_GID_INDEX=3       # Use RoCE v2 GID
export NCCL_SOCKET_IFNAME=eth0   # Use this interface
export NCCL_IB_HCA=mlx5_0        # Use this RDMA device

EOF
