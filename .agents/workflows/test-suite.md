### /test-suite

Execute the full edge AI test suite across all subsystems:

1. **Step 1: Configuration Validation**
   Run configuration integrity tests:
   ```bash
   pytest tests/test_skills_config.py -v
   ```

2. **Step 2: Model & Architecture Invariants**
   Verify static input shapes, channel protection, and graph constraints:
   ```bash
   pytest tests/test_model_invariants.py -v
   ```

3. **Step 3: Export & Quantization Validation**
   Validate ONNX graphs and QDQ compliance:
   ```bash
   pytest tests/test_onnx_export.py -v
   ```

4. **Step 4: Benchmark Telemetry Schema**
   Verify benchmark CSV logging schema and metric calculations:
   ```bash
   pytest tests/test_logging_schema.py -v
   ```

5. **Step 5: Review & Sign-Off**
   Report aggregated test results and request human review before proceeding to deployment or physical hardware benchmarking.
