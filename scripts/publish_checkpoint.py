#!/usr/bin/env python3
"""
Publish / retrieve a fine-tuned X-VC checkpoint to a Hugging Face model repo.

Checkpoints are large binaries and must NOT be committed to git (see .gitignore).
Instead keep the durable copy in a model store so it survives container resets,
fresh clones, and branch switches. This helper is a thin wrapper around
`huggingface_hub` for exactly that.

    # one-time: authenticate (writes a token to ~/.cache/huggingface)
    huggingface-cli login          # or: export HF_TOKEN=hf_xxx

    # link repo + token once via the scripts/finetune.sh prompt (saved to the
    # gitignored .hf_publish.env, auto-read here), or set them per shell:
    #   export XVC_PUBLISH_REPO=<user>/<repo>   # else pass --repo-id each time

    # push the best checkpoint of a run as "finetune_arabic.pt"
    python scripts/publish_checkpoint.py push \
        --repo-id your-org/xvc-accent-finetunes --private \
        --checkpoint exp/finetune_arabic/ckpt/000750.pt \
        --name finetune_arabic.pt \
        --config exp/finetune_arabic/config.yaml

    # pull it back on the serving container
    python scripts/publish_checkpoint.py pull \
        --repo-id your-org/xvc-accent-finetunes \
        --name finetune_arabic.pt \
        --out ckpts/finetune_arabic.pt

`huggingface_hub` ships with `transformers` (already in requirements.txt); a token
with write access is needed for `push`. Part of the X-VC accent fine-tuning pipeline
(upstream: https://github.com/Jerrister/X-VC, MIT).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _require_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "       It ships with transformers (requirements.txt); on a bare env run:\n"
            "         pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return huggingface_hub


def _load_publish_env() -> None:
    """Read repo-root .hf_publish.env (written by the finetune.sh link prompt).

    Simple KEY=VALUE lines; already-set environment variables always win.
    """
    env_file = Path(__file__).resolve().parents[1] / ".hf_publish.env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _token() -> Optional[str]:
    # Explicit env wins; otherwise huggingface_hub falls back to the cached login.
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def do_push(args) -> int:
    hub = _require_hub()
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    remote_name = args.name or ckpt.name
    token = _token()

    size = ckpt.stat().st_size
    digest = _sha256(ckpt)
    print(f"checkpoint : {ckpt}")
    print(f"size       : {_human_size(size)}")
    print(f"sha256     : {digest}")
    print(f"repo       : {args.repo_id} (private={args.private})")
    print(f"remote name: {remote_name}")

    hub.create_repo(
        repo_id=args.repo_id, repo_type="model", private=args.private,
        exist_ok=True, token=token,
    )
    url = hub.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=remote_name,
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Add {remote_name} (sha256={digest[:12]})",
    )
    print(f"uploaded   : {url}")

    if args.config:
        cfg = Path(args.config).resolve()
        if not cfg.is_file():
            print(f"WARNING: config not found, skipping: {cfg}", file=sys.stderr)
        else:
            cfg_remote = remote_name.rsplit(".", 1)[0] + ".yaml"
            cfg_url = hub.upload_file(
                path_or_fileobj=str(cfg),
                path_in_repo=cfg_remote,
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Add {cfg_remote}",
            )
            print(f"config     : {cfg_url}")

    # Provenance sidecars: run_meta.json / eval summary.csv travel WITH the
    # weights, under the checkpoint's basename, so the frozen checkpoint is
    # forever traceable to its code, data, and selection curve.
    stem = remote_name.rsplit(".", 1)[0]
    extras = list(args.extra or [])
    if args.run_dir:
        for candidate in ("run_meta.json", os.path.join("eval", "summary.csv")):
            p = Path(args.run_dir) / candidate
            if p.is_file():
                extras.append(str(p))
            else:
                print(f"WARNING: --run-dir sidecar not found, skipping: {p}", file=sys.stderr)
    for extra in extras:
        p = Path(extra).resolve()
        if not p.is_file():
            print(f"WARNING: extra not found, skipping: {p}", file=sys.stderr)
            continue
        extra_remote = f"{stem}.{p.name}"
        url = hub.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=extra_remote,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Add {extra_remote}",
        )
        print(f"extra      : {url}")

    print("\nPUSH OK")
    return 0


def do_pull(args) -> int:
    hub = _require_hub()
    token = _token()
    out = Path(args.out).resolve() if args.out else Path(args.name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"repo       : {args.repo_id}")
    print(f"remote name: {args.name}")
    downloaded = hub.hf_hub_download(
        repo_id=args.repo_id,
        filename=args.name,
        repo_type="model",
        token=token,
        revision=args.revision,
    )
    # hf_hub_download returns a cache path; copy to the requested location.
    import shutil
    shutil.copyfile(downloaded, out)
    digest = _sha256(out)
    print(f"saved to   : {out}")
    print(f"size       : {_human_size(out.stat().st_size)}")
    print(f"sha256     : {digest}")
    print("\nPULL OK (verify sha256 against the value printed at push time)")
    return 0


def main(argv=None) -> int:
    _load_publish_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    default_repo = os.environ.get("XVC_PUBLISH_REPO")
    if default_repo:
        # tolerate URL-ish values: leading/trailing slashes, full hf.co URLs, quotes
        default_repo = default_repo.strip().strip("'\"")
        default_repo = default_repo.removeprefix("https://huggingface.co/")
        default_repo = default_repo.strip("/") or None

    p = sub.add_parser("push", help="upload a checkpoint to a HF model repo")
    p.add_argument("--repo-id", required=default_repo is None, default=default_repo,
                   help="HF model repo (default: $XVC_PUBLISH_REPO / .hf_publish.env)")
    p.add_argument("--checkpoint", required=True, help="local .pt path to upload")
    p.add_argument("--name", default=None, help="remote filename (default: basename of --checkpoint)")
    p.add_argument("--config", default=None, help="optional config.yaml to upload alongside")
    p.add_argument("--run-dir", default=None,
                   help="training run dir; auto-uploads run_meta.json and eval/summary.csv")
    p.add_argument("--extra", action="append", default=None,
                   help="additional sidecar file to upload (repeatable)")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.set_defaults(func=do_push)

    g = sub.add_parser("pull", help="download a checkpoint from a HF model repo")
    g.add_argument("--repo-id", required=default_repo is None, default=default_repo,
                   help="HF model repo (default: $XVC_PUBLISH_REPO / .hf_publish.env)")
    g.add_argument("--name", required=True, help="remote filename to download")
    g.add_argument("--out", default=None, help="local destination path (default: ./<name>)")
    g.add_argument("--revision", default=None, help="branch/tag/commit (default: main)")
    g.set_defaults(func=do_pull)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
