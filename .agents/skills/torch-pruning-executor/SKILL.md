---
name: torch-pruning-executor
description: Applies dependency-graph structured channel pruning to YOLOX models using Torch-Pruning (DepGraph). Trigger this skill when performing structural channel pruning, filter reduction, or MAC reduction on PyTorch checkpoints.
---

# Torch-Pruning Structured Channel Pruning Skill

When invoked:
1. Parse the target sparsity ratio from `.agents/skills.config.yaml` (`pruning.default_sparsity`).
2. Load the trained PyTorch model and trace graph dependencies using `tp.DependencyGraph()`.
3. Protect the YOLOX detection heads from channel removal:
   - Identify all final `Conv2d` layers where `out_channels == 85` (COCO 80 classes + 4 bbox + 1 obj) and add them to `ignored_layers`.
4. Apply structured pruning using `tp.pruner.MagnitudePruner` with L1-norm importance (`tp.importance.MagnitudeImportance(p=1)`).
5. **CRITICAL**: Save the entire PyTorch model object via `torch.save(model, path)` instead of saving `state_dict`, because the layer channel dimensions are physically modified.
6. Execute a 20-epoch fine-tuning run on the COCO training set with learning rate scaled to 0.1x of baseline.