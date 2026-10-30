"""Stable sample identities, run validation and atomic outputs for inference."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sample_id(item):
    """Use an explicit dataset ID, or hash the input path and edit pair."""
    if "id" in item:
        if not isinstance(item["id"], (str, int)) or isinstance(item["id"], bool):
            raise ValueError("Sample id must be a string or integer")
        identity = {"id": item["id"]}
    else:
        identity = {key: item[key] for key in ("image_path", "text_s", "text_t")}
        identity["image_path"] = os.path.normpath(identity["image_path"]).replace("\\", "/")
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def sample_seed(base_seed, item):
    """Independent of sample order, resume filtering and distributed rank."""
    digest = hashlib.sha256(f"{int(base_seed)}:{sample_id(item)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def output_path(output_dir, item):
    stem = Path(item["image_path"].replace("\\", "/")).stem
    stem = re.sub(r"[^\w.-]", "_", stem)[:64] or "image"
    return str(Path(output_dir) / f"{stem}_{sample_id(item)}.png")


def is_output_complete(path):
    """A filename alone is insufficient evidence of a completed image."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
        return False


def atomic_save_image(image, path):
    """Only expose the final PNG after Pillow has finished writing it."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".image-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_args(args):
    """Reject incompatible CLI options before loading model weights."""
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if args.use_att and args.scheduler != "overshoot":
        raise ValueError("Attention requires --scheduler overshoot; use --no-use_att with euler")
    if getattr(args, "use_flowedit", False):
        if not 0 <= args.flowedit_start_step < args.flowedit_end_step <= args.num_inference_steps:
            raise ValueError("FlowEdit requires 0 <= start < end <= num_inference_steps")
        if args.n_avg < 1:
            raise ValueError("n_avg must be positive")


def prepare_run(output_dir, options, entry, data=None):
    """All ranks validate the same immutable configuration before inference."""
    options = dict(options) if isinstance(options, dict) else vars(options).copy()
    if data is None:
        with open(options["json_path"], encoding="utf-8") as stream:
            data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of records")
    seen = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) for key in ("image_path", "text_s", "text_t")
        ):
            raise ValueError(f"Invalid dataset record at index {index}")
        identity = sample_id(item)
        if identity in seen:
            raise ValueError(f"Duplicate sample identity at index {index}; supply unique id fields")
        seen.add(identity)

    # Locations and GPU count may change on resume; effective inference options may not.
    options.pop("exp_dir", None)
    options.pop("json_path", None)
    config = {
        "schema_version": 1,
        "entry": Path(entry).name,
        "options": options,
        "dataset_sha256": hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "run_config.json"
    if not manifest.exists() and any(output_dir.glob("*.png")):
        raise ValueError("Existing images have no run_config.json; choose a new experiment directory")
    fd, temporary = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(config) + "\n")
        # A hard link publishes complete JSON without overwriting another rank's manifest.
        try:
            os.link(temporary, manifest)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary)
    with manifest.open(encoding="utf-8") as stream:
        existing = json.load(stream)
    if existing != config:
        raise ValueError("Run configuration or dataset differs; choose a new experiment directory")
    return data
