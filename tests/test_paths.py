"""default_still prefers demo_rocket when present."""

from pathlib import Path

from scripts._paths import ROOT, default_still


def test_default_still_skips_phone_gallery_when_demo_exists(tmp_path, monkeypatch):
    # Exercise real helper against repo layout when demo_rocket exists
    still = default_still()
    assert still.exists()
    assert still.name != "IMG_0026.png"
    assert still.suffix.lower() in {".jpg", ".jpeg", ".png"}
