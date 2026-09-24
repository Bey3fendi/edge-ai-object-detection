---
name: edge-test-harness
description: Executes unit and integration test suites using pytest, verifying edge AI hardware constraints, ONNX graph sanity, DepGraph pruning safety, QDQ quantization integrity, and benchmark telemetry formatting.
---

# Edge Test Harness Skill

When invoked:

1. **Environment Setup & Discovery**:
   - Discover all test modules inside `tests/`.
   - Ensure the active Python environment has `pytest` installed.
   
2. **Execute Test Categories**:
   - **Configuration Tests**: Validate `.agents/skills.config.yaml` schema and values.
   - **Model & Pruning Invariants**: Validate fixed `[1, 3, 416, 416]` input tensors, output head protection (`out_channels == 85`), and DepGraph structural integrity.
   - **Export & Quantization Tests**: Verify `onnx.checker` integrity, opset version, and QDQ node insertion for INT8 models.
   - **Telemetry Schema Tests**: Verify `logs/benchmark_results.csv` exact column headers and type safety.

3. **Standard Test Execution Command**:
   ```bash
   pytest -v tests/ --tb=short
   ```

4. **Failure Triage & Human-in-the-Loop Reporting**:
   - If tests fail, summarize the failure trace with exact file links and line numbers.
   - Propose the minimal remediation to the user and request review before making changes.
