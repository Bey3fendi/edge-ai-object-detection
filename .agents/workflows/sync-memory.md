### /sync-memory

Autonomously synchronizes project status, updates decisions, archives legacy items, and verifies Hafıza OS consistency:

1. **Step 1: Inspect Project State**
   Scan recent git commits, exported artifacts in `artifacts/`, and benchmark entries in `logs/benchmark_results.csv`.

2. **Step 2: Update Project DURUM.md**
   - Automatically update `/home/furkan/Hafiza/projeler/tez-edge-yolox/DURUM.md`.
   - Update **"Nerede Kaldı"** with completed items.
   - Update **"Sıradaki Adım"** with next priority tasks.
   - Move deprecated decisions to **"Eski / Değişen Kararlar Arşivi (Legacy)"**.

3. **Step 3: Verify Memory Integrity**
   Run Hafıza OS validation:
   ```bash
   python3 -X utf8 /home/furkan/Hafiza/araclar/hafiza.py --vault /home/furkan/Hafiza validate
   ```

4. **Step 4: Report Status**
   Present a 3-bullet summary to Furkan of what was synchronized, with zero manual input required from him.
