# models/

Downloaded model weights are cached here by `huggingface_hub` and are **never
committed to Git** (see `.gitignore`).

Default model: [`adeIaide/skin-lesion-7class-efficientnet`](https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet)
pinned to commit `b5ebe73c3b89baa90ee4f991afa318385414a973`.

| File | Size | Purpose |
|---|---|---|
| `model.safetensors` | 16.3 MB | EfficientNet-B0 weights (tensor-only, safe to load) |
| `config.json` | 1 KB | class order, input size, normalisation |

Download with:

```bash
uv run skin-research download-model
```

Licence: CC-BY-NC-4.0 (non-commercial research and education only, inherited
from the ISIC 2018 / HAM10000 training data). See `docs/MODEL_AND_DATASET.md`.
