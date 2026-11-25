"""Flask application: page routes plus the JSON API consumed by the dashboard."""

from __future__ import annotations

import os
import tempfile

from flask import (Flask, jsonify, render_template, request, send_file, abort)

from ..engine import CorrectionConfig
from ..engine.occurrence import OCC_DETERMINISTIC, OCC_STOCHASTIC
from ..engine.amount import TAIL_CLIP, TAIL_EXTRAPOLATE
from . import state


def create_app(runtime_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    runtime_dir = runtime_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    state.init_store(os.path.join(runtime_dir, "calibrations"))
    app.config["RUNTIME_DIR"] = runtime_dir

    # ---------------------------------------------------------------- pages #
    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------ API #
    @app.post("/api/data/synthetic")
    def api_synthetic():
        manifest = state.load_synthetic()
        return jsonify(state._to_jsonable(manifest))

    @app.post("/api/data/import")
    def api_import():
        up_h = request.files.get("history")
        up_f = request.files.get("future")
        if not up_h or not up_f:
            abort(400, "需要上传 history 与 future 两个 CSV")
        with tempfile.TemporaryDirectory() as tmp:
            hp = os.path.join(tmp, "history.csv")
            fp = os.path.join(tmp, "future.csv")
            up_h.save(hp)
            up_f.save(fp)
            manifest = state.load_imported(hp, fp)
        return jsonify(state._to_jsonable(manifest))

    @app.post("/api/data/revise")
    def api_revise():
        multiplier = float(request.json.get("multiplier", 1.1))
        addend = float(request.json.get("addend", 0.0))
        new_rev = state.revise_observations(multiplier, addend)
        return jsonify({"obs_revision": new_rev,
                        "calibrations": state.list_calibrations()})

    @app.get("/api/data/summary")
    def api_summary():
        st = state.STATE
        if st.history is None:
            return jsonify({"loaded": False})
        return jsonify(state._to_jsonable({
            "loaded": True,
            "source": st.source,
            "manifest": st.manifest,
            "obs_revision": st.obs_revision,
            "calibrations": state.list_calibrations(),
        }))

    def _config_from_request(payload) -> CorrectionConfig:
        return CorrectionConfig(
            season_mode=payload.get("season_mode", "seasonal"),
            occ_method=payload.get("occ_method", "auto"),
            amount_method=payload.get("amount_method", "auto"),
            occ_mode=payload.get("occ_mode", OCC_DETERMINISTIC),
            tail=payload.get("tail", TAIL_EXTRAPOLATE),
            tail_quantile=float(payload.get("tail_quantile", 0.85)),
            train_years=int(payload.get("train_years", 6)),
            val_years=int(payload.get("val_years", 3)),
            test_years=int(payload.get("test_years", 3)),
            stride=int(payload.get("stride", 1)),
            seed=payload.get("seed"),
        )

    @app.post("/api/cv")
    def api_cv():
        payload = request.get_json(force=True)
        config = _config_from_request(payload)
        return jsonify(state.run_cv(config))

    @app.post("/api/calibrate")
    def api_calibrate():
        payload = request.get_json(force=True)
        config = _config_from_request(payload)
        occ_mode = payload.get("occ_mode", OCC_DETERMINISTIC)
        tail = payload.get("tail", TAIL_EXTRAPOLATE)
        seed = payload.get("seed")
        use_rec = bool(payload.get("use_recommendation", True))
        return jsonify(state.calibrate_and_apply(
            config, occ_mode, tail, seed, use_rec))

    @app.post("/api/compare")
    def api_compare():
        payload = request.get_json(force=True)
        config = _config_from_request(payload)
        tail = payload.get("tail", TAIL_EXTRAPOLATE)
        occ_mode = payload.get("occ_mode", OCC_DETERMINISTIC)
        season = payload.get("season", "JJA")
        return jsonify(state.compare_methods(
            tail, occ_mode, season, config))

    @app.get("/api/explain")
    def api_explain():
        cal_id = request.args["calibration_id"]
        row = int(request.args["row_index"])
        return jsonify(state.explain(cal_id, row))

    @app.get("/api/calibrations")
    def api_calibrations():
        return jsonify(state.list_calibrations())

    @app.get("/api/export")
    def api_export():
        cal_id = request.args["calibration_id"]
        path = state.export(cal_id, os.path.join(
            app.config["RUNTIME_DIR"], f"{cal_id}_product.csv"))
        return send_file(path, as_attachment=True, download_name=
                         f"{cal_id}_corrected.csv")

    @app.errorhandler(ValueError)
    def handle_value_error(err):
        return jsonify({"error": str(err)}), 400

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(err):
        return jsonify({"error": str(err)}), 409

    return app
