---
description: Enforces fully autonomous memory lifecycle maintenance. The user shall never edit memory files manually.
alwaysOn: true
---

# Autonomous Memory Lifecycle Rules

1. **Zero Manual Burden on User**:
   Furkan shall NEVER be asked to manually edit, update, format, or clean up memory files (`DURUM.md`, `çekirdek.md`, `hafiza.py`, etc.). All memory maintenance is the sole responsibility of the Agent.

2. **Automatic Milestone & State Sync**:
   Whenever a major milestone is reached (e.g. baseline evaluated, pruning round completed, INT8 model exported, benchmark executed):
   - The Agent must automatically update `/home/furkan/Hafiza/projeler/tez-edge-yolox/DURUM.md`.
   - Update **"Nerede Kaldı"** with the completed work and timestamp.
   - Update **"Sıradaki Adım"** with the concrete next action.

3. **Legacy Decision Archival (Deprecation Protocol)**:
   Whenever an architectural choice or parameter is superseded or changed:
   - Do NOT simply delete it.
   - The Agent must autonomously move the outdated item to **"7. Eski / Değişen Kararlar Arşivi (Legacy)"** in `DURUM.md` with the date, the reason for the change, and the new replacement decision.
   - Update **"2. Kilitli Kararlar"** with the new binding decision.

4. **Session Boundary Preservation**:
   At the end of significant conversation phases or when switching tasks, the Agent verifies that changes are recorded so that subsequent sessions pick up immediately via the `PreInvocation` hook without asking Furkan for context.
