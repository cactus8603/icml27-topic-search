"""[codex] Extract only inference tensors from the two author Drive releases.

The original upload SHA and source file ID remain in the receipt. Only the
known NumPy scalar/dtype globals required for old training counters are allowed;
arbitrary pickle execution is never enabled. Each saved tensor is compared
exactly before removing our own downloaded optimizer-bearing upload.
"""
from pathlib import Path
import hashlib
import json
import os

PINS = {
    "negclip-original.pth": {"file_id":"1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", "bytes":1815619309,
        "source_sha256":"f50ae34b6c311670b1bd3c072d266d27c8750aad5e24b8e2b0862d941f681ffd"},
    "triplet-openclip-finetune.pt": {"file_id":"14mupW26LMh6U4FQPa74FOIMEg8MndxCh", "bytes":1815698002,
        "source_sha256":"1d828e66d9906ac71aa7054ed9b695f44b38052f4258a951681de43afe73ae54"},
}


def main():
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    root = Path(__file__).resolve().parents[2]/"dl/retrieval-matrix-extra/weights"
    receipt = root/"conversion-receipt.json"
    output = json.loads(receipt.read_text()) if receipt.exists() else {}
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
               np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
    for name, pin in PINS.items():
        file_id = pin["file_id"]
        source = root/name
        if name in output and not source.exists():
            row = output[name]
            with Path(row["path"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
            continue
        with source.open("rb") as stream:
            original_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if original_sha != pin["source_sha256"] or source.stat().st_size != pin["bytes"]:
            raise ValueError("Mutable Drive upload differs from the preregistered original payload SHA/size")
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError("Author checkpoint did not contain a tensor-only model state")
        prefix = "module." if all(key.startswith("module.") for key in state) else ""
        tensors = {key.removeprefix(prefix): value.contiguous() for key, value in state.items()}
        target = source.with_suffix(".safetensors")
        save_file(tensors, str(target), metadata={"source_sha256": original_sha,
                  "source_file_id": file_id, "conversion": "tensor-only; uniform module. prefix removed if present"})
        loaded = load_file(target)
        if loaded.keys() != tensors.keys() or any(not torch.equal(loaded[key], value) for key, value in tensors.items()):
            raise ValueError("Tensor extraction changed original model values")
        with target.open("rb") as stream:
            target_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        output[name] = {"source_file_id": file_id, "source_url": f"https://drive.google.com/file/d/{file_id}/view",
            "source_sha256": original_sha, "source_bytes": source.stat().st_size,
            "path": str(target), "sha256": target_sha, "bytes": target.stat().st_size,
            "keys": len(tensors), "uniform_prefix_removed": prefix, "every_tensor_equal": True,
            "omitted_training_fields": [key for key in checkpoint if key not in ("state_dict", "model")],
            "author_source": "negclip_original" if name.startswith("neg") else "tripletclip",
            "revision_note": "Drive has no immutable public revision; author repository commit + file ID + exact payload SHA identify this release",
            "raw_removed": False}
        receipt.write_text(json.dumps(output, indent=2)+"\n")
        if source.stat().st_uid != os.getuid():
            raise ValueError("Cannot remove a download owned by another user")
        del loaded, tensors, state, checkpoint
        source.unlink()
        output[name]["raw_removed"] = True
        receipt.write_text(json.dumps(output, indent=2)+"\n")
        print(name, target.stat().st_size, target_sha, flush=True)


if __name__ == "__main__":
    main()
