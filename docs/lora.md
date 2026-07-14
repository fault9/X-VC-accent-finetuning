# LoRA adapters

The implementation is custom (no PEFT), small, and checkpoint-compatible by
construction. Engine: `xvc/adapters/lora.py` (the historical import path
`models.codec.sac.modules.lora` re-exports it). API: `xvc.adapters`.

## Design contract

* `LoRALinear` **subclasses** `nn.Linear`: the base `weight`/`bias` keep their
  original state-dict keys. A stock checkpoint loads into a LoRA-injected
  model by exact key match; only `lora_A`/`lora_B` are new keys.
* `lora_B` is zero-initialized → a freshly injected adapter is the identity;
  warm-starting reproduces stock behavior until training moves `lora_B`.
* Forward: `y = W x + b + (alpha / r) * B A x` (dropout on the adapter input).
* `merge()` folds `scaling * B @ A` into `weight` (idempotent); `unmerge()`
  subtracts it back — lossless up to float addition. After merging, forward
  is a plain linear (zero adapter overhead).

## Where LoRA is applied

Injection targets `nn.Linear` layers by **substring matching** on qualified
module names, scoped to top-level submodules:

| Config (adapter group) | `target_modules` | `include` |
|---|---|---|
| `adapter/lora_acoustic` | `[acoustic_converter]` | `attn.`, `ff_x.ff`, `ff_c.ff` |
| `adapter/lora_acoustic_prenet` | `+ prenet` | `+ pwconv` |
| `adapter/lora_distill_stack` | `+ prenet` | `+ attn_norm_x.`, `norm_out.` (AdaLN) |

`"attn."` covers both attention streams — including the target-conditioning
projections `to_q_c/to_k_c/to_v_c/to_out_c` — so the frame-level reference
path IS adapted. The converter's final block has no `ff_c`/`to_out_c` (it
does not update the conditioning stream): with depth *d*, the converter set
adapts `12(d-1) + 9` layers.

**AdaLN-Zero subtlety:** block outputs are gated by AdaLN projections that
are zero-initialized in a fresh model. In a *randomly initialized* model the
gates are 0 and adapter deltas cannot reach the output (the smoke tests wake
the gates explicitly). Warm-started models have trained, non-zero gates —
this never affects real experiments, but will bite anyone unit-testing
against a fresh model.

## API

```python
from xvc.adapters import (
    inject_lora,                      # keyword API; returns InjectionReport
    freeze_all_parameters,
    unfreeze_lora_parameters,
    get_trainable_parameter_report,   # fails loudly if nothing is trainable
    merge_lora_weights,
    load_lora_state_dict,             # adapter tensors only, old layouts OK
)

report = inject_lora(
    generator,
    rank=4, alpha=16, dropout=0.0,
    include_patterns=["attn.", "ff_x.ff", "ff_c.ff"],
    target_modules=["acoustic_converter"],
)
print(report.adapted_layers)          # every adapted layer, deterministic order
```

Failure modes are typed and loud: `NoTargetsMatchedError` when the filters
match nothing (a config that would otherwise "train" zero adapters),
`DuplicateInjectionError` when a layer is already wrapped.

Training-side injection happens in `utils/train_utils.py::maybe_inject_lora`
**before** the warm-start load (so base weights land in the frozen base);
inference-side in `XVC.load_from_checkpoint` (so checkpoint adapter keys
match). Freezing goes through `xvc.adapters.freezing`; the startup gate
`verify_trainable_modules` aborts on any mismatch.

## Adding a new LoRA target

1. Find the layer names: `[n for n, m in generator.named_modules()
   if isinstance(m, torch.nn.Linear)]`.
2. Add a new adapter group under `configs/adapter/` with the include set, and
   make sure `trainable_modules` lists every host submodule (the freeze gate
   verifies against it).
3. Dry-run: `python scripts/train.py experiment=<...> dry_run=true`, then
   check the injection report in the startup log lists exactly the layers you
   expect.
4. After training, run `scripts/verify_lora_hygiene.py --config ... --ckpt ...
   --stock ckpts/xvc.pt` — it proves the freeze held (bitwise) and merge
   equivalence on the real checkpoint.

## Deployment

`scripts/merge_lora.py` exports a merged, stock-architecture checkpoint
(optionally `--lora-scale`), servable with zero LoRA code. LoRA-only
checkpoints (adapters without the base, ~1000x smaller) can be written and
inspected via `xvc.training.checkpointing` — see `docs/checkpoints.md`.
