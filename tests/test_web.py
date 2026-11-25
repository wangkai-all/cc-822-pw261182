"""Flask route smoke tests: pages, static assets, full API flow, import path."""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app.web.app import create_app
from app.web import state


@pytest.fixture()
def client(tmp_path):
    app = create_app(runtime_dir=str(tmp_path / "runtime"))
    app.config.update(TESTING=True)
    # Reset the singleton for an isolated test.
    state.STATE.history = None
    state.STATE.future = None
    state.STATE.calibrations.clear()
    state.STATE.products.clear()
    return app.test_client()


def _load_synthetic(client):
    return client.post("/api/data/synthetic")


def test_index_and_assets_render(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "降水预报偏差校正" in html
    for asset in ("/static/styles.css", "/static/charts.js", "/static/app.js"):
        assert client.get(asset).status_code == 200


def test_full_api_workflow(client):
    assert _load_synthetic(client).status_code == 200
    summary = client.get("/api/data/summary").get_json()
    assert summary["loaded"] is True

    cv = client.post("/api/cv", json={
        "season_mode": "seasonal", "train_years": 6,
        "val_years": 3, "test_years": 3}).get_json()
    assert len(cv["folds"]) == 7

    cal = client.post("/api/calibrate", json={
        "season_mode": "seasonal", "occ_method": "auto", "amount_method": "auto",
        "occ_mode": "stochastic", "tail": "extrapolate", "seed": 7,
        "use_recommendation": True,
        "train_years": 6, "val_years": 3, "test_years": 3}).get_json()
    cid = cal["calibration_id"]

    compare = client.post("/api/compare", json={
        "season_mode": "seasonal", "season": "JJA",
        "tail": "extrapolate", "occ_mode": "deterministic",
        "train_years": 6, "val_years": 3, "test_years": 3}).get_json()
    assert set(compare["series"]) == {"plain_qm", "trend_qdm"}
    # Both families expose the wet-frequency and quantile diagnostics.
    for fam in compare["series"].values():
        assert "wet_frequency" in fam and "quantiles" in fam

    explain = client.get(f"/api/explain?calibration_id={cid}&row_index=0").get_json()
    assert "occurrence" in explain and "binding" in explain

    export = client.get(f"/api/export?calibration_id={cid}")
    assert export.status_code == 200
    assert b"forecast_raw" in export.data and b"corrected" in export.data


def test_revision_marks_old_product_stale(client):
    _load_synthetic(client)
    client.post("/api/calibrate", json={
        "occ_method": "trend", "amount_method": "qdm",
        "occ_mode": "deterministic", "use_recommendation": False,
        "train_years": 6, "val_years": 3, "test_years": 3})
    before = client.get("/api/calibrations").get_json()
    assert before[0]["binding"]["status"] == "current"
    client.post("/api/data/revise", json={"multiplier": 1.1})
    after = client.get("/api/calibrations").get_json()
    assert after[0]["binding"]["status"] == "stale"
    # The old calibration id is retained unchanged.
    assert after[0]["calibration_id"] == before[0]["calibration_id"]


def test_csv_import_endpoint(client, history, future):
    h_buf = io.BytesIO()
    f_buf = io.BytesIO()
    history[["date", "obs", "hindcast"]].to_csv(h_buf, index=False)
    future[["date", "forecast"]].to_csv(f_buf, index=False)
    h_buf.seek(0); f_buf.seek(0)
    resp = client.post("/api/data/import", data={
        "history": (h_buf, "history.csv"),
        "future": (f_buf, "future.csv"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["history_years"][0] == int(history["year"].min())
    # The imported data is now the active dataset.
    summary = client.get("/api/data/summary").get_json()
    assert summary["loaded"] is True and summary["source"] == "imported"


def test_action_without_data_is_409(client):
    resp = client.post("/api/cv", json={"train_years": 6, "val_years": 3,
                                        "test_years": 3})
    assert resp.status_code == 409
