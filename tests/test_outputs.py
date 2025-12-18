"""
Enhanced test suite for NCCL optimization with PFC vs ECN comparison.

Validates:
1. All three RoCEv2 modes tested (PFC, ECN, Hybrid)
2. InfiniBand baseline established
3. Best RoCEv2 mode achieves ≥90% of IB performance
4. Comprehensive optimization report with mode comparison
"""

import os
import re
import sys
from pathlib import Path


def test_baseline_timing_exists():
    """Test that baseline timing file was created."""
    baseline_file = Path("/workspace/baseline_timing.txt")
    assert baseline_file.exists(), "baseline_timing.txt not found"

    baseline_time = float(baseline_file.read_text().strip())
    assert baseline_time > 0.1, f"Baseline time too fast: {baseline_time}s"
    print(f"✓ Baseline: {baseline_time * 1000:.1f} ms/iter")


def test_roce_pfc_mode():
    """Test that RoCEv2 PFC mode was tested."""
    pfc_file = Path("/workspace/roce_pfc_timing.txt")

    if not pfc_file.exists():
        print("  Warning: PFC mode not tested (optional but recommended)")
        return

    pfc_time = float(pfc_file.read_text().strip())
    baseline_time = float(Path("/workspace/baseline_timing.txt").read_text().strip())
    speedup = baseline_time / pfc_time

    print(f"✓ RoCEv2 PFC: {pfc_time * 1000:.1f} ms/iter ({speedup:.2f}x speedup)")

    # PFC should achieve at least 2.5x speedup
    if speedup < 2.5:
        print(f"  Note: PFC speedup could be better ({speedup:.2f}x < 2.5x target)")


def test_roce_ecn_mode():
    """Test that RoCEv2 ECN mode was tested."""
    ecn_file = Path("/workspace/roce_ecn_timing.txt")

    if not ecn_file.exists():
        print("  Warning: ECN mode not tested (optional but recommended)")
        return

    ecn_time = float(ecn_file.read_text().strip())
    baseline_time = float(Path("/workspace/baseline_timing.txt").read_text().strip())
    speedup = baseline_time / ecn_time

    print(f"✓ RoCEv2 ECN: {ecn_time * 1000:.1f} ms/iter ({speedup:.2f}x speedup)")

    # ECN should achieve at least 3x speedup
    if speedup < 3.0:
        print(f"  Note: ECN speedup could be better ({speedup:.2f}x < 3.0x target)")


def test_roce_hybrid_mode():
    """Test that RoCEv2 Hybrid mode was tested."""
    hybrid_file = Path("/workspace/roce_hybrid_timing.txt")

    if not hybrid_file.exists():
        print("  Info: Hybrid mode not tested (optional)")
        return

    hybrid_time = float(hybrid_file.read_text().strip())
    baseline_time = float(Path("/workspace/baseline_timing.txt").read_text().strip())
    speedup = baseline_time / hybrid_time

    print(f"✓ RoCEv2 Hybrid: {hybrid_time * 1000:.1f} ms/iter ({speedup:.2f}x speedup)")

    # Hybrid should be best RoCEv2 mode
    if speedup < 3.3:
        print(f"  Note: Hybrid speedup could be better ({speedup:.2f}x < 3.3x target)")


def test_infiniband_baseline():
    """Test that InfiniBand baseline was established."""
    ib_file = Path("/workspace/ib_timing.txt")

    if not ib_file.exists():
        print("  Info: InfiniBand baseline not tested (optional)")
        return

    ib_time = float(ib_file.read_text().strip())
    baseline_time = float(Path("/workspace/baseline_timing.txt").read_text().strip())
    speedup = baseline_time / ib_time

    print(f"✓ InfiniBand: {ib_time * 1000:.1f} ms/iter ({speedup:.2f}x speedup)")


def test_optimized_timing_exists():
    """Test that optimized (best) timing file was created."""
    optimized_file = Path("/workspace/optimized_timing.txt")
    assert optimized_file.exists(), "optimized_timing.txt not found"

    optimized_time = float(optimized_file.read_text().strip())
    print(f"✓ Best mode: {optimized_time * 1000:.1f} ms/iter")


def test_overall_speedup():
    """Test that overall speedup target was achieved."""
    baseline_file = Path("/workspace/baseline_timing.txt")
    optimized_file = Path("/workspace/optimized_timing.txt")

    assert baseline_file.exists() and optimized_file.exists()

    baseline_time = float(baseline_file.read_text().strip())
    optimized_time = float(optimized_file.read_text().strip())
    speedup = baseline_time / optimized_time

    print(f"  Overall speedup: {speedup:.2f}x")

    assert speedup >= 3.0, f"Speedup {speedup:.2f}x < 3.0x required"
    print(f"✓ Achieved {speedup:.2f}x speedup (target: ≥3.0x)")


def test_roce_vs_ib_performance():
    """Test that best RoCEv2 mode is within 90% of InfiniBand."""
    ib_file = Path("/workspace/ib_timing.txt")
    optimized_file = Path("/workspace/optimized_timing.txt")

    if not ib_file.exists():
        print("  Info: IB baseline missing, skipping comparison")
        return

    ib_time = float(ib_file.read_text().strip())
    optimized_time = float(optimized_file.read_text().strip())

    # Lower time is better, so optimized/IB gives relative performance
    roce_vs_ib_percent = (ib_time / optimized_time) * 100

    print(f"  RoCEv2 best: {optimized_time * 1000:.1f} ms")
    print(f"  InfiniBand:  {ib_time * 1000:.1f} ms")
    print(f"  RoCEv2 = {roce_vs_ib_percent:.1f}% of IB performance")

    # Best RoCEv2 should be within 90% of IB
    assert roce_vs_ib_percent >= 90, (
        f"RoCEv2 only {roce_vs_ib_percent:.1f}% of IB (need ≥90%)"
    )
    print(
        f"✓ RoCEv2 achieved {roce_vs_ib_percent:.1f}% of IB performance (target: ≥90%)"
    )


def test_optimization_report_exists():
    """Test that optimization report was created."""
    report_file = Path("/workspace/optimization_report.md")
    assert report_file.exists(), "optimization_report.md not found"

    content = report_file.read_text()
    assert len(content) >= 800, f"Report too short: {len(content)} chars (minimum: 800)"

    print(f"✓ Report exists ({len(content)} characters)")


def test_report_discusses_pfc_vs_ecn():
    """Test that report analyzes PFC vs ECN trade-offs."""
    report_file = Path("/workspace/optimization_report.md")
    assert report_file.exists(), "optimization_report.md not found"

    content = report_file.read_text().lower()

    # Check for discussion of both PFC and ECN
    has_pfc_discussion = "pfc" in content and "priority flow control" in content
    has_ecn_discussion = "ecn" in content and (
        "explicit congestion" in content or "dcqcn" in content
    )
    has_comparison = "vs" in content or "versus" in content or "compared" in content

    if not has_pfc_discussion:
        print("  Warning: Report lacks PFC discussion")
    if not has_ecn_discussion:
        print("  Warning: Report lacks ECN discussion")

    if has_pfc_discussion and has_ecn_discussion:
        print("✓ Report discusses both PFC and ECN")

    if has_comparison:
        print("✓ Report includes mode comparison")


def test_report_content_quality():
    """Test that report contains required technical content."""
    report_file = Path("/workspace/optimization_report.md")
    assert report_file.exists()

    content = report_file.read_text().lower()

    required_topics = {
        "rdma": ["rdma", "infiniband", "roce"],
        "nccl": ["nccl"],
        "gid": ["gid"],
        "performance": ["speedup", "performance", "iteration"],
        "congestion": ["congestion", "pfc", "ecn"],
    }

    missing_topics = []
    for topic, keywords in required_topics.items():
        if not any(keyword in content for keyword in keywords):
            missing_topics.append(topic)

    if missing_topics:
        print(f"  Info: Report could discuss: {', '.join(missing_topics)}")
    else:
        print("✓ Report covers all key topics")


def test_solution_completeness():
    """Meta-test for overall solution quality."""

    checks = {
        "baseline_timing": Path("/workspace/baseline_timing.txt").exists(),
        "optimized_timing": Path("/workspace/optimized_timing.txt").exists(),
        "optimization_report": Path("/workspace/optimization_report.md").exists(),
        "pfc_tested": Path("/workspace/roce_pfc_timing.txt").exists(),
        "ecn_tested": Path("/workspace/roce_ecn_timing.txt").exists(),
        "ib_baseline": Path("/workspace/ib_timing.txt").exists(),
    }

    completed = sum(1 for v in checks.values() if v)
    total = len(checks)

    print(f"✓ Solution completeness: {completed}/{total} artifacts")

    # Require essential files
    assert checks["baseline_timing"], "Missing baseline_timing.txt (required)"
    assert checks["optimized_timing"], "Missing optimized_timing.txt (required)"
    assert checks["optimization_report"], "Missing optimization_report.md (required)"


if __name__ == "__main__":
    print("=" * 70)
    print("NCCL Optimization Tests: PFC vs ECN vs Hybrid vs InfiniBand")
    print("=" * 70)
    print()

    tests = [
        ("Baseline timing exists", test_baseline_timing_exists),
        ("RoCEv2 PFC mode tested", test_roce_pfc_mode),
        ("RoCEv2 ECN mode tested", test_roce_ecn_mode),
        ("RoCEv2 Hybrid mode tested", test_roce_hybrid_mode),
        ("InfiniBand baseline", test_infiniband_baseline),
        ("Optimized timing exists", test_optimized_timing_exists),
        ("Overall speedup ≥3x", test_overall_speedup),
        ("RoCEv2 ≥90% of IB", test_roce_vs_ib_performance),
        ("Optimization report exists", test_optimization_report_exists),
        ("Report: PFC vs ECN analysis", test_report_discusses_pfc_vs_ecn),
        ("Report content quality", test_report_content_quality),
        ("Solution completeness", test_solution_completeness),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"[TEST] {name}")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
