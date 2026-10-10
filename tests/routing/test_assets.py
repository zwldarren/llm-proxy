from llm_proxy.routing import assets


def test_packaged_assets_present():
    base = assets._package_assets_dir()
    assert (base / "model.json").is_file()
    assert (base / "calibration_params.json").is_file()


def test_ensure_assets_deployed_copies_to_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_data_dir", lambda: tmp_path)
    deployed = assets.ensure_assets_deployed()
    assert (deployed / "model.json").is_file()
    assert (deployed / "calibration_params.json").is_file()
