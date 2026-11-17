"""Bind the reverse Mistral model roles to the verified forward model identities."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scd_runtime.contracts.identity import canonical_bytes
from scd_runtime.contracts.types import profile_from_dict, to_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_profile", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    forward = profile_from_dict(json.loads(args.forward_profile.read_bytes()))
    identity_map = tuple(range(forward.source.num_layers))
    if (forward.reader.kind != "none" or forward.layer_map != identity_map
            or forward.source.attention["sliding_window"] is not None
            or forward.target.attention["sliding_window"] != 4096):
        raise ValueError("expected the verified MistralLite to Mistral-7B-v0.1 profile")
    reverse = replace(
        forward, profile_id="icml_mistral_reverse_cmrc", source=forward.target, target=forward.source,
        reader=replace(forward.reader, base_manifest_sha256=forward.source.manifest_sha256),
        asset_manifest_sha256="0" * 64, validation_protocol=None,
    )
    with args.out.open("xb") as stream:
        stream.write(canonical_bytes(to_data(reverse)))


if __name__ == "__main__":
    main()
