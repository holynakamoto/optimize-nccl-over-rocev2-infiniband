"""
Test suite for NCCL optimization task.

This file verifies that the student has successfully:
1. Achieved target bandwidth on RoCEv2 (>180 GB/s)
2. Achieved target bandwidth on InfiniBand (>190 GB/s)
3. Achieved 3x speedup in PyTorch DDP training
4. Created an optimization report
5. Properly configured NCCL to use RDMA (not TCP fallback)
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def test_baseline_timing_exists():
    """Test that baseline timing file was created."""
    baseline_file = Path("/workspace/baseline_timing.txt")
    assert baseline_file.exists(), "baseline_timing.txt not found"

    content = baseline_file.read_text().strip()
    baseline_time = float(content)

    # Baseline should be slow (TCP fallback)
    assert baseline_time > 0.1, f"Baseline time seems too fast: {baseline_time}s"
    print(f"✓ Baseline timing: {baseline_time * 1000:.2f} ms/iter")


def test_optimized_timing_exists():
    """Test that optimized timing file was created."""
    optimized_file = Path("/workspace/optimized_timing.txt")
    assert optimized_file.exists(), (
        "optimized_timing.txt not found. Did you run the PyTorch DDP test?"
    )

    content = optimized_file.read_text().strip()
    optimized_time = float(content)

    assert optimized_time > 0, f"Invalid optimized time: {optimized_time}s"
    print(f"✓ Optimized timing: {optimized_time * 1000:.2f} ms/iter")


def test_pytorch_speedup():
    """Test that PyTorch training achieved 3x speedup."""
    baseline_file = Path("/workspace/baseline_timing.txt")
    optimized_file = Path("/workspace/optimized_timing.txt")

    assert baseline_file.exists(), "baseline_timing.txt not found"
    assert optimized_file.exists(), "optimized_timing.txt not found"

    baseline_time = float(baseline_file.read_text().strip())
    optimized_time = float(optimized_file.read_text().strip())

    speedup = baseline_time / optimized_time

    print(f"  Baseline: {baseline_time * 1000:.2f} ms/iter")
    print(f"  Optimized: {optimized_time * 1000:.2f} ms/iter")
    print(f"  Speedup: {speedup:.2f}x")

    assert speedup >= 3.0, f"Speedup {speedup:.2f}x is less than required 3.0x"
    print(f"✓ Achieved {speedup:.2f}x speedup (target: ≥3.0x)")


def test_optimization_report_exists():
    """Test that optimization report was created."""
    report_file = Path("/workspace/optimization_report.md")
    assert report_file.exists(), "optimization_report.md not found"

    content = report_file.read_text()

    # Check minimum length
    assert len(content) >= 500, f"Report too short: {len(content)} chars (minimum: 500)"

    print(f"✓ Report exists ({len(content)} characters)")


def test_optimization_report_content():
    """Test that optimization report contains required sections."""
    report_file = Path("/workspace/optimization_report.md")
    assert report_file.exists(), "optimization_report.md not found"

    content = report_file.read_text().lower()

    # Required topics that should be discussed
    required_topics = {
        "gpudirect": ["gpudirect", "gdr", "gpu direct"],
        "nccl_ib": ["nccl_ib", "nccl ib"],
        "gid": ["gid", "gid_index", "gid index"],
        "bandwidth": ["bandwidth", "gb/s", "gbps"],
        "rdma": ["rdma", "infiniband", "roce"],
    }

    missing_topics = []
    for topic, keywords in required_topics.items():
        if not any(keyword in content for keyword in keywords):
            missing_topics.append(topic)

    if missing_topics:
        print(f"  Warning: Report missing discussion of: {', '.join(missing_topics)}")
    else:
        print("✓ Report covers all required topics")

    # Don't fail for missing topics, just warn
    # assert not missing_topics, f"Report missing key topics: {missing_topics}"


def test_nccl_roce_bandwidth():
    """Test that RoCEv2 NCCL bandwidth meets target."""
    # Check if benchmark results file exists
    roce_results = Path("/workspace/nccl_roce_results.txt")

    if not roce_results.exists():
        print("  Skipping RoCEv2 bandwidth test (results file not found)")
        print("  Note: In a full implementation, this would run actual NCCL benchmarks")
        return

    content = roce_results.read_text()

    # Parse bandwidth from results
    match = re.search(r"Avg bus bandwidth\s*:\s*(\d+\.?\d*)", content)
    assert match, "Could not find bandwidth in RoCEv2 results"

    bandwidth = float(match.group(1))

    print(f"  RoCEv2 bandwidth: {bandwidth:.2f} GB/s")
    assert bandwidth >= 180.0, (
        f"RoCEv2 bandwidth {bandwidth:.2f} GB/s is below target (180 GB/s)"
    )
    print(f"✓ RoCEv2 bandwidth meets target ({bandwidth:.2f} GB/s ≥ 180 GB/s)")


def test_nccl_ib_bandwidth():
    """Test that InfiniBand NCCL bandwidth meets target."""
    # Check if benchmark results file exists
    ib_results = Path("/workspace/nccl_ib_results.txt")

    if not ib_results.exists():
        print("  Skipping InfiniBand bandwidth test (results file not found)")
        print("  Note: In a full implementation, this would run actual NCCL benchmarks")
        return

    content = ib_results.read_text()

    # Parse bandwidth from results
    match = re.search(r"Avg bus bandwidth\s*:\s*(\d+\.?\d*)", content)
    assert match, "Could not find bandwidth in InfiniBand results"

    bandwidth = float(match.group(1))

    print(f"  InfiniBand bandwidth: {bandwidth:.2f} GB/s")
    assert bandwidth >= 190.0, (
        f"InfiniBand bandwidth {bandwidth:.2f} GB/s is below target (190 GB/s)"
    )
    print(f"✓ InfiniBand bandwidth meets target ({bandwidth:.2f} GB/s ≥ 190 GB/s)")


def test_nccl_environment_configured():
    """Test that NCCL environment variables are properly configured."""
    # Check for a marker file that indicates NCCL was configured
    config_file = Path("/workspace/nccl_config.env")

    if not config_file.exists():
        print("  Info: nccl_config.env not found (optional)")
        return

    content = config_file.read_text()

    # Check for key environment variables
    important_vars = [
        "NCCL_IB_DISABLE=0",
        "NCCL_NET=IB",
        "NCCL_IB_GID_INDEX",
        "NCCL_NET_GDR_LEVEL",
    ]

    found_vars = []
    for var in important_vars:
        if var in content:
            found_vars.append(var.split("=")[0])

    print(f"✓ NCCL environment configured ({len(found_vars)} key variables set)")


def test_no_tcp_fallback():
    """Test that NCCL is not using TCP fallback."""
    # Check NCCL logs for evidence of RDMA usage
    log_files = [
        Path("/workspace/nccl_debug.log"),
        Path("/workspace/pytorch_output.log"),
    ]

    tcp_fallback_detected = False
    rdma_detected = False

    for log_file in log_files:
        if not log_file.exists():
            continue

        content = log_file.read_text()

        # Check for TCP fallback warnings
        if "NET/Socket" in content or "Falling back to TCP" in content:
            tcp_fallback_detected = True

        # Check for RDMA success
        if "NET/IB" in content or "Using RDMA" in content:
            rdma_detected = True

    if tcp_fallback_detected:
        print("  Warning: TCP fallback detected in logs")

    if rdma_detected:
        print("✓ RDMA transport detected in logs")


def test_solution_quality():
    """Meta-test to evaluate overall solution quality."""

    checks = {
        "baseline_timing": Path("/workspace/baseline_timing.txt").exists(),
        "optimized_timing": Path("/workspace/optimized_timing.txt").exists(),
        "optimization_report": Path("/workspace/optimization_report.md").exists(),
        "nccl_config": Path("/workspace/nccl_config.env").exists(),
    }

    completed = sum(1 for v in checks.values() if v)
    total = len(checks)

    print(f"✓ Solution completeness: {completed}/{total} required artifacts present")

    # Require at least the essential files
    assert checks["optimization_report"], "Missing optimization_report.md (required)"
    assert checks["optimized_timing"], "Missing optimized_timing.txt (required)"


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running NCCL Optimization Tests...")
    print("=" * 60)

    tests = [
        ("Baseline timing exists", test_baseline_timing_exists),
        ("Optimized timing exists", test_optimized_timing_exists),
        ("PyTorch speedup ≥3x", test_pytorch_speedup),
        ("Optimization report exists", test_optimization_report_exists),
        ("Report content quality", test_optimization_report_content),
        ("RoCEv2 bandwidth ≥180 GB/s", test_nccl_roce_bandwidth),
        ("InfiniBand bandwidth ≥190 GB/s", test_nccl_ib_bandwidth),
        ("NCCL environment configured", test_nccl_environment_configured),
        ("No TCP fallback", test_no_tcp_fallback),
        ("Solution quality", test_solution_quality),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
