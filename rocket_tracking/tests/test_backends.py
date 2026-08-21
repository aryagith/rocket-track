"""Backend availability helpers."""

from rocket_track.backends import list_backend_availability


def test_list_backend_availability_includes_cpu():
    infos = {i.name: i for i in list_backend_availability()}
    assert "pytorch_cpu" in infos
    assert infos["pytorch_cpu"].available is True
    assert "tensorrt" in infos
