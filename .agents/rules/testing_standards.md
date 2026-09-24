---
description: Universal testing standards requiring unit and integration tests for all project code and artifacts.
alwaysOn: true
---

# Edge AI Unit & Integration Testing Standards

1. **Mandatory Test-First or Test-With Code**:
   Every new script, model transformation, export pipeline, or telemetry utility must have corresponding unit tests in `tests/` before it is marked complete.

2. **Semi-Automated Human Review Gate**:
   All test runs, code changes, and bug fixes proposed by testing subagents must be presented to the user for explicit review and approval (`commandExecutionPolicy: ask`). No automated bypasses are permitted.

3. **Fast Offline Execution**:
   Unit tests must run fast and without internet connectivity using synthetic tensors (e.g. `[1, 3, 416, 416]`) and small mock fixtures. Do not download or rely on full COCO datasets for basic unit tests.

4. **Edge AI Invariant Assertions**:
   - **Static Tensor Constraint**: Validate that model inputs are strictly static `(1, 3, H, W)` and reject dynamic axes.
   - **Detection Head Protection**: Assert that final `Conv2d` layers with 85 output channels are never pruned or structurally altered.
   - **Graph Sanity**: Validate all exported graphs using `onnx.checker.check_model`.
   - **Quantization Verification**: Confirm presence of QDQ (`QuantizeLinear` and `DequantizeLinear`) nodes in quantized graphs.
   - **Telemetry Schema Validation**: Verify that `logs/benchmark_results.csv` adheres strictly to the mandated 24-column header schema.
