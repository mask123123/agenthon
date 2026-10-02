"""Release consistency: the descriptor discloses the exact coefs.json that ships in the image."""
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_descriptor_discloses_shipped_coefs():
    sha = hashlib.sha256((ROOT / "textlayer" / "coefs.json").read_bytes()).hexdigest()
    src = (ROOT / "make_descriptor.py").read_text()
    assert re.search(r"sha256:" + sha, src), "refit coefs.json -> update the sha256 in make_descriptor.py MODELS"


def test_dockerfile_ships_textlayer():
    assert "COPY textlayer /opt/textlayer" in (ROOT / "Dockerfile").read_text()


if __name__ == "__main__":
    test_descriptor_discloses_shipped_coefs()
    test_dockerfile_ships_textlayer()
    print("PASS release consistency")
