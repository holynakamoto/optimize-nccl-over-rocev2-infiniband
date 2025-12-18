#!/bin/bash
# Baseline NCCL benchmark script
# This will show poor performance due to TCP fallback

set -e

echo "=============================================="
echo "NCCL Baseline Benchmark"
echo "=============================================="
echo ""
echo "Environment:"
env | grep NCCL | sort
echo ""
echo "Running all_reduce_perf test..."
echo "This simulates 4 GPUs performing collective communication"
echo ""

cd /opt/nccl-tests

# Simulate the benchmark output with poor TCP performance
# In a real environment, this would actually run the MPI command
# mpirun -np 4 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

cat << 'EOF'
# nThread 1 nGpus 1 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12345 on  localhost device  0 [0x00] NVIDIA A100-SXM4-40GB
#  Rank  1 Group  0 Pid  12346 on  localhost device  1 [0x00] NVIDIA A100-SXM4-40GB
#  Rank  2 Group  0 Pid  12347 on  localhost device  2 [0x00] NVIDIA A100-SXM4-40GB
#  Rank  3 Group  0 Pid  12348 on  localhost device  3 [0x00] NVIDIA A100-SXM4-40GB
#
# Using NCCL version 2.22.3
# NCCL_NET: Socket
# WARN: Falling back to TCP/IP socket transport
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    45.21    0.00    0.00      0    44.89    0.00    0.00      0
          16             4     float     sum      -1    45.67    0.00    0.00      0    45.23    0.00    0.00      0
          32             8     float     sum      -1    46.12    0.00    0.00      0    45.78    0.00    0.00      0
          64            16     float     sum      -1    47.34    0.00    0.00      0    46.91    0.00    0.00      0
         128            32     float     sum      -1    49.56    0.00    0.00      0    48.23    0.00    0.00      0
         256            64     float     sum      -1    52.78    0.00    0.01      0    51.45    0.00    0.01      0
         512           128     float     sum      -1    58.90    0.01    0.01      0    57.23    0.01    0.01      0
        1024           256     float     sum      -1    67.12    0.02    0.02      0    65.67    0.02    0.02      0
        2048           512     float     sum      -1    82.45    0.02    0.04      0    80.12    0.03    0.04      0
        4096          1024     float     sum      -1    112.3    0.04    0.06      0    109.8    0.04    0.06      0
        8192          2048     float     sum      -1    167.8    0.05    0.07      0    163.4    0.05    0.08      0
       16384          4096     float     sum      -1    278.9    0.06    0.09      0    271.2    0.06    0.09      0
       32768          8192     float     sum      -1    498.7    0.07    0.10      0    485.3    0.07    0.10      0
       65536         16384     float     sum      -1    923.4    0.07    0.11      0    897.8    0.07    0.11      0
      131072         32768     float     sum      -1   1789.2    0.07    0.11      0   1742.3    0.08    0.11      0
      262144         65536     float     sum      -1   3456.7    0.08    0.11      0   3367.1    0.08    0.12      0
      524288        131072     float     sum      -1   6712.3    0.08    0.12      0   6534.8    0.08    0.12      0
     1048576        262144     float     sum      -1   13234    0.08    0.12      0   12876     0.08    0.12      0
     2097152        524288     float     sum      -1   26123    0.08    0.12      0   25432     0.08    0.12      0
     4194304       1048576     float     sum      -1   51876    0.08    0.12      0   50543     0.08    0.13      0
     8388608       2097152     float     sum      -1   103234   0.08    0.13      0   100567    0.08    0.13      0
    16777216       4194304     float     sum      -1   205678   0.08    0.13      0   200345    0.08    0.13      0
    33554432       8388608     float     sum      -1   410987   0.08    0.13      0   400234    0.08    0.13      0
    67108864      16777216     float     sum      -1   821345   0.08    0.13      0   800123    0.08    0.13      0
   134217728      33554432     float     sum      -1   1642345  0.08    0.13      0   1600234   0.08    0.13      0
   268435456      67108864     float     sum      -1   3284567  0.08    0.13      0   3200345   0.08    0.13      0
   536870912     134217728     float     sum      -1   6569012  0.08    0.13      0   6400567   0.08    0.13      0
  1073741824     268435456     float     sum      -1   13138023 0.08    0.13      0   12801123  0.08    0.13      0
  2147483648     536870912     float     sum      -1   26276045 0.08    0.13      0   25602245  0.08    0.13      0
  4294967296    1073741824     float     sum      -1   52552089 0.08    0.13      0   51204489  0.08    0.13      0
  8589934592    2147483648     float     sum      -1   105104178 0.08   0.13      0   102408978 0.08    0.13      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.08
#

EOF

echo ""
echo "=============================================="
echo "Baseline Performance Summary"
echo "=============================================="
echo "Average bus bandwidth: ~0.08 GB/s"
echo "Transport used: TCP/IP Socket (SLOW!)"
echo ""
echo "This is MUCH slower than expected!"
echo "Expected: 180+ GB/s with optimized RDMA"
echo ""
echo "Your task: Fix the configuration to enable RDMA"
echo "=============================================="
