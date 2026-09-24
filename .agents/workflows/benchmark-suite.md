### /benchmark-suite

Execute the full two-stage benchmark protocol across connected edge targets:

1. **Step 1: Environmental Check**
   Verify hardware governor and active cooling states on both devices.
2. **Step 2: Stage-1 Portable CPU Baseline**
   Run `scripts/evaluation/eval_ort_cpu.py` for 5,000 COCO val iterations (after 100 warmup iterations) on both Raspberry Pi 5 and Jetson Orin Nano.
3. **Step 3: Stage-2 Optimized Runs**
   - On Raspberry Pi 5: Run `artifacts/ncnn/yolox.param` via ncnn benchmark harness.
   - On Jetson Orin Nano: Run `artifacts/tensorrt/yolox_int8.plan` via `trtexec`.
4. **Step 4: Telemetry Aggregation**
   Trigger the `telemetry-analyzer` skill to parse power logs, calculate $FPS/W$ and $J/frame$, and append all records to `logs/benchmark_results.csv`.