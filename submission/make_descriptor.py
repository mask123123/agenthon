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
a = ap.parse_args()
d = {
    "schema_version": "1.1.0", "interface_version": "2.0",
    "competition_id": "agenthon2026-forecasting-dev", "team_id": a.team_id,
    "track": "forecasting", "phase": "dev", "category": "api",
    "image": {"registry": a.registry, "repository": a.repo, "digest": a.digest},
    "image_access": "public", "models": [], "license": "MIT",
    "descriptor_digest": "sha256:" + "0" * 64,
}
d = seal_descriptor_digest(d)
SubmissionDescriptor.from_mapping(d)           # raises if invalid
json.dump(d, open(a.out, "w"), indent=2)
print("wrote", a.out, "valid; descriptor_digest", d["descriptor_digest"])
