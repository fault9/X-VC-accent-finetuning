"""X-VC accent-finetuning library package.

Import-light by design: importing ``xvc`` (or any of its submodules that unit
tests rely on) must not pull in deepspeed, wandb, matplotlib, or audiotools.
Heavyweight dependencies are imported inside the functions that need them.
"""

__version__ = "0.1.0"
