# Bundled models

Drop fastdet models (`.fdt` files, https://github.com/PCJohn/fastdet) here and the gate
loads them by file name: `text.fdt` replaces the heuristic text map with the model's
probabilities, and any other name (`face.fdt`, `person.fdt`, ...) becomes
`FrameStats.model_maps[name]`. A model must be trained on the gate's front-end (the
default `GateConfig`: 1024-px box-filtered thumbnail, stride 1, the 64..2 grid pyramid);
the gate checks and refuses a mismatch. `GateConfig(models={"text": path})` points at a
file elsewhere, `GateConfig(models={"text": None})` skips a bundled one. Without fastdet
installed, bundled models are skipped with a warning and the heuristics stay in use.
