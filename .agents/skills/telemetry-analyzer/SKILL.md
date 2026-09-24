---
name: telemetry-analyzer
description: Parses execution logs from tegrastats, Linux pidstat, and external power analyzers to generate Pareto analysis. Trigger this skill when computing FPS/Watt, Joule/frame, or plotting Pareto efficiency charts.
---

# Hardware Telemetry Analysis Skill

When processing telemetry:
1. Parse Jetson power rails from `logs/telemetry/tegrastats_load.log` (extract `VDD_IN` average mW).
2. Parse Raspberry Pi 5 external USB-C power readings.
3. Compute efficiency metrics:
   $$\text{FPS/W} = \frac{\text{Throughput (FPS)}}{P_{load} \text{ (Watts)}}$$
   $$\text{Joule/frame} = \frac{P_{load} \text{ (Watts)}}{\text{Throughput (FPS)}}$$
4. Format comparative output tables and append results to `logs/benchmark_results.csv`.