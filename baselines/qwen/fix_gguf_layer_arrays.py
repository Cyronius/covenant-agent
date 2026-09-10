"""Fix a latent bug in the NextN metadata patch (train_s3.sh, train_s4.sh):
`gguf_set_metadata ... qwen35.block_count 24 --force` drops the NextN/MTP
prediction layer from the block count, but doesn't touch any per-layer
array field -- `qwen35.attention.recurrent_layers` stays the original
25-entry array. Older llama.cpp didn't validate array length against
block_count and loaded it anyway (S1-S3's GGUFs predate this field
entirely -- a still-older converter never emitted it). A current build
does validate and refuses to load: "key qwen35.attention.recurrent_layers
has wrong array length; expected 24, got 24+1".

Rebuilds the file with every array field of length block_count+1 truncated
to block_count entries, dropping the trailing ones -- the NextN layer is
always appended after the real stack, never inserted in the middle.

  python fix_gguf_layer_arrays.py IN.gguf OUT.gguf
"""
from __future__ import annotations

import argparse

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def native(x):
    return x.item() if hasattr(x, "item") else x


def block_count(r: GGUFReader) -> int:
    for f in r.fields.values():
        if f.name.endswith(".block_count"):
            return native(f.parts[f.data[0]][0])
    raise SystemExit("no *.block_count field in this file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    r = GGUFReader(args.src)
    n = block_count(r)
    arch_field = r.fields["general.architecture"]
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode()

    writer = GGUFWriter(args.dst, arch=arch)  # sets general.architecture itself
    writer.data_alignment = r.alignment

    fixed = []
    for f in r.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue
        vtype = f.types[0]
        if vtype == GGUFValueType.ARRAY:
            subtype = f.types[1]
            if subtype == GGUFValueType.STRING:
                vals = [bytes(f.parts[i]).decode("utf-8", errors="replace")
                        for i in f.data]
            else:
                vals = [native(f.parts[i][0]) for i in f.data]
            # a per-layer array one longer than block_count carries the
            # bolted-on NextN layer at the end -- drop it
            if len(vals) == n + 1:
                vals = vals[:n]
                fixed.append(f.name)
            writer.add_array(f.name, vals)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(f.name, bytes(f.parts[f.data[0]]).decode(
                "utf-8", errors="replace"))
        else:
            writer.add_key_value(f.name, native(f.parts[f.data[0]][0]), vtype)

    for t in r.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"fixed {fixed or '(nothing needed fixing)'} -> {args.dst}")


if __name__ == "__main__":
    main()
