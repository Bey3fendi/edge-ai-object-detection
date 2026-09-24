#### 3. `.agents/rules/benchmark_logging_rules.md`
*Activation: Always On*

```markdown
---
description: Mandates single-source-of-truth telemetry recording for benchmark runs.
alwaysOn: true
---

# Benchmark Telemetry Logging Rules

All hardware execution scripts must write output directly to `logs/benchmark_results.csv`.
If the file does not exist, create it with this exact header schema:

```csv
run_id,device,device_mode,model_variant,backend,precision,input_size,batch,threads,map5095,ap50,latency_mean_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,fps,power_idle_w,power_load_w,power_dynamic_w,fps_per_watt,joule_per_frame,peak_ram_mb,temperature_start,temperature_end