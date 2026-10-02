"""Executive summary: write a valid submission.json for the dev phase from (digest, team_id), validate it
with the toolkit's own parser, and reseal descriptor_digest. Usage:
  python make_descriptor.py --digest sha256:<64hex> --team-id team-<32hex> [--repo mask123123/t2-forecaster]
The Team Key is never needed here: team_id is printed by `qfbench2 submission alias`."""
import argparse, json
from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest

ap = argparse.ArgumentParser()
ap.add_argument("--digest", required=True)
ap.add_argument("--team-id", required=True)
ap.add_argument("--repo", default="mask123123/t2-forecaster")
ap.add_argument("--registry", default="ghcr.io")
ap.add_argument("--out", default="submission.json")
ap.add_argument("--models", choices=["textlayer", "house", "none"], default="textlayer",
                help="what the image can use: textlayer = House model + fitted Fed-tone logistic (this tree), house = House model only (numeric image with the verified number-extraction fallback), none = no model")
ap.add_argument("--house", action="store_true", help="alias for --models house")
a = ap.parse_args()
# House model (textlayer/fed_tone.py) and the fitted logistic shipped as textlayer/coefs.json (artifact policy:
# bundled fitted models are disclosed with access "local" and their training cutoff). tests/test_release.py checks the sha.
MODELS = [
    {"name": "nvidia/nemotron-3-super-120b-a12b", "version": "rl-030326-fp8", "revision": "rl-030326-fp8",
     "training_cutoff": "unpublished", "access": "api"},
    {"name": "team304/ust-fed-tone-logit", "version": "v1",
     "revision": "sha256:452207a16c33227c88159dcf21eaa7b6f51aeacb7dfedf3026b8aaafd00dc132", "training_cutoff": "2024-12-18", "access": "local"},
]
d = {
    "schema_version": "1.1.0", "interface_version": "2.0",
    "competition_id": "agenthon2026-forecasting-dev", "team_id": a.team_id,
    "track": "forecasting", "phase": "dev", "category": "api",
    "image": {"registry": a.registry, "repository": a.repo, "digest": a.digest},
    "image_access": "public", "models": MODELS if a.models == "textlayer" and not a.house else (MODELS[:1] if (a.house or a.models == "house") else []), "license": "MIT",
    "descriptor_digest": "sha256:" + "0" * 64,
}
d = seal_descriptor_digest(d)
SubmissionDescriptor.from_mapping(d)           # raises if invalid
json.dump(d, open(a.out, "w"), indent=2)
print("wrote", a.out, "valid; descriptor_digest", d["descriptor_digest"])
