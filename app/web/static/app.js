/* Main dashboard controller. Talks to the Flask JSON API and renders all views. */
(function () {
  const $ = (id) => document.getElementById(id);
  const fmt3 = (d) => (d === null || d === undefined || !isFinite(d)) ? "—" : (+d).toFixed(3);
  const fmt2 = (d) => (d === null || d === undefined || !isFinite(d)) ? "—" : (+d).toFixed(2);

  let lastCompare = null;
  let historyFile = null, futureFile = null;

  function showError(msg) {
    const box = $("errorBox");
    box.textContent = msg; box.hidden = false;
    setTimeout(() => { box.hidden = true; }, 6000);
  }
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }
  function post(path, body) {
    return api(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  function collectConfig() {
    const seedVal = $("seedInput").value;
    return {
      season_mode: $("seasonMode").value,
      occ_method: $("occMethod").value,
      amount_method: $("amountMethod").value,
      occ_mode: $("occMode").value,
      tail: $("tailMode").value,
      tail_quantile: parseFloat($("tailQuantile").value),
      train_years: parseInt($("trainYears").value, 10),
      val_years: parseInt($("valYears").value, 10),
      test_years: parseInt($("testYears").value, 10),
      stride: parseInt($("stride").value, 10),
      seed: seedVal === "" ? null : parseInt(seedVal, 10),
    };
  }

  // --------------------------------------------------------------- data load
  async function refreshSummary() {
    const s = await api("/api/data/summary");
    if (!s.loaded) {
      $("dataBadge").textContent = "未载入数据";
      $("dataBadge").className = "badge badge-muted";
      $("revBadge").hidden = true;
      return;
    }
    $("dataBadge").textContent = `已载入 · ${s.source}`;
    $("dataBadge").className = "badge badge-good";
    $("revBadge").hidden = false;
    $("revBadge").textContent = `观测修订 ${s.obs_revision}`;
    $("revBadge").className = "badge badge-warn";
    renderManifest(s.manifest);
    renderCalibrationList(s.calibrations);
  }

  function renderManifest(m) {
    if (!m) { $("manifest").hidden = true; return; }
    $("manifest").hidden = false;
    const bias = m.known_bias
      ? `毛毛雨概率 ${m.known_bias.drizzle_prob} · 强度倍率 ${m.known_bias.intensity_multiplier}`
      : "（导入数据，无预设偏差声明）";
    const trend = m.known_trend
      ? `湿日log-odds/年 ${m.known_trend.wet_logodds_per_year} · 强度log/年 ${m.known_trend.intensity_log_per_year}`
      : "（导入数据，无预设趋势声明）";
    $("manifest").innerHTML = `
      <span>历史年份</span><b>${m.history_years[0]}–${m.history_years[1]}</b>
      <span>未来年份</span><b>${m.future_years[0]}–${m.future_years[1]}</b>
      <span>历史湿日比例 · 观测</span><b>${fmt3(m.history_wet_freq_obs)}</b>
      <span>历史湿日比例 · 历史预报</span><b>${fmt3(m.history_wet_freq_hindcast)}</b>
      <span>已知偏差</span><b>${bias}</b>
      <span>已知趋势</span><b>${trend}</b>`;
  }

  // --------------------------------------------------------------- compare
  async function runCompare() {
    const cfg = collectConfig();
    const data = await post("/api/compare", { ...cfg, season: $("seasonSelect").value });
    lastCompare = data;
    renderWetFreq(data);
    renderQuantiles(data);
    renderTail(data);
    renderSeries(data);
  }

  function seriesColors() { return Charts.SERIES(); }

  function renderWetFreq(data) {
    const [c1, c2, c3] = seriesColors();
    const truthWf = data.truth ? data.truth.wet_frequency : null;
    const groups = ["plain_qm", "trend_qdm"].map((name) => {
      const s = data.series[name];
      const values = [
        { name: "原始预报", value: s.wet_frequency.raw },
        { name: "校正后", value: s.wet_frequency.corrected },
      ];
      if (truthWf !== null) values.push({ name: "未来真值", value: truthWf });
      return { label: name === "plain_qm" ? "普通 QM" : "趋势 QDM", values };
    });
    const ret = (name) => {
      const r = data.series[name].trend_retention;
      return r === undefined ? "—" : fmt3(r);
    };
    const cards = $("wetFreqCards");
    cards.innerHTML = "";
    const truthLine = truthWf !== null
      ? `<div class="stat"><div class="k"><span class="swatch" style="background:${c3}"></span>未来真值湿日比例</div><div class="v">${fmt3(truthWf)}</div></div>` : "";
    cards.innerHTML = `
      <div class="stat"><div class="k"><span class="swatch" style="background:${c1}"></span>原始预报（两族相同）</div><div class="v">${fmt3(data.series.plain_qm.wet_frequency.raw)}</div></div>
      <div class="stat"><div class="k"><span class="swatch" style="background:${c2}"></span>普通 QM 校正</div><div class="v">${fmt3(data.series.plain_qm.wet_frequency.corrected)}</div><div class="k" style="margin-top:4px">趋势信号保留度 ${ret("plain_qm")}</div></div>
      <div class="stat"><div class="k"><span class="swatch" style="background:${c2}"></span>趋势 QDM 校正</div><div class="v">${fmt3(data.series.trend_qdm.wet_frequency.corrected)}</div><div class="k" style="margin-top:4px">趋势信号保留度 ${ret("trend_qdm")}</div></div>
      ${truthLine}`;
    Charts.barChart($("wetFreqChart"), { groups });
    Charts.legend($("wetFreqChart"), groups[0].values.map((v) => ({ name: v.name })));
  }

  function renderQuantiles(data) {
    const [c1, c2, c3] = seriesColors();
    const probs = data.series.plain_qm.quantiles.probs.map((p) => `q${p}`);
    const cleanSeries = [
      { name: "原始预报", color: c1,
        points: data.series.plain_qm.quantiles.raw.map((y) => ({ x: 0, y })) },
      { name: "普通 QM 校正", color: c2,
        points: data.series.plain_qm.quantiles.corrected.map((y) => ({ x: 0, y })) },
      { name: "趋势 QDM 校正", color: c3,
        points: data.series.trend_qdm.quantiles.corrected.map((y) => ({ x: 0, y })) },
    ];
    if (data.truth) cleanSeries.push({
      name: "未来真值", color: cssMuted(),
      points: data.truth.quantiles.map((y) => ({ x: 0, y }))
    });
    Charts.lineChart($("quantileChart"), {
      series: cleanSeries, xLabels: probs, height: 340,
      xTitle: "经验分位（仅湿日量级）", yTitle: "降水量 mm", yFmt: fmt2
    });
    // Table view (accessibility + exact numbers).
    const q = data.series;
    let rows = q.plain_qm.quantiles.probs.map((p, i) => `
      <tr><td>q${p}</td>
        <td>${fmt2(q.plain_qm.quantiles.raw[i])}</td>
        <td>${fmt2(q.plain_qm.quantiles.corrected[i])}</td>
        <td>${fmt2(q.trend_qdm.quantiles.corrected[i])}</td>
        <td>${data.truth ? fmt2(data.truth.quantiles[i]) : "—"}</td></tr>`).join("");
    $("quantileTable").innerHTML = `<table>
      <thead><tr><th>分位</th><th>原始</th><th>QM 校正</th><th>QDM 校正</th><th>真值</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }
  function cssMuted() {
    return getComputedStyle(document.querySelector(".viz-root")).getPropertyValue("--muted").trim();
  }

  function renderTail(data) {
    const [c1, c2, c3] = seriesColors();
    const tQ = data.series.plain_qm.tail_corrected.quantiles;
    const cards = $("tailCards");
    const extrapP = data.series.plain_qm.n_extrapolated;
    const extrapT = data.series.trend_qdm.n_extrapolated;
    cards.innerHTML = `
      <div class="stat"><div class="k">尾部模式</div><div class="v" style="font-size:16px">${data.tail_mode === "clip" ? "截断 clip" : "GPD 外推"}</div></div>
      <div class="stat"><div class="k">QM 超训练范围点数</div><div class="v">${extrapP}</div></div>
      <div class="stat"><div class="k">QDM 超训练范围点数</div><div class="v">${extrapT}</div></div>
      <div class="stat"><div class="k"><span class="swatch" style="background:${c2}"></span>QM 最大值</div><div class="v">${fmt2(data.series.plain_qm.tail_corrected.max)}<small> mm</small></div></div>
      <div class="stat"><div class="k"><span class="swatch" style="background:${c2}"></span>QDM 最大值</div><div class="v">${fmt2(data.series.trend_qdm.tail_corrected.max)}<small> mm</small></div></div>
      ${data.truth ? `<div class="stat"><div class="k"><span class="swatch" style="background:${c3}"></span>真值最大值</div><div class="v">${fmt2(data.truth.tail.max)}<small> mm</small></div></div>` : ""}`;
    const labels = tQ.map((p) => `q${p}`);
    const tailSeries = [
      { name: "QM 校正尾部", color: c2,
        points: data.series.plain_qm.tail_corrected.values.map((y) => ({ x: 0, y })) },
      { name: "QDM 校正尾部", color: c3,
        points: data.series.trend_qdm.tail_corrected.values.map((y) => ({ x: 0, y })) },
    ];
    if (data.truth) tailSeries.push({
      name: "真值尾部", color: c1,
      points: data.truth.tail.values.map((y) => ({ x: 0, y }))
    });
    Charts.lineChart($("tailChart"), {
      series: tailSeries, xLabels: labels, height: 260,
      xTitle: "极高分位", yTitle: "降水量 mm", yFmt: fmt2
    });
  }

  function renderSeries(data) {
    // Thin out to keep the SVG readable; show the QDM corrected vs raw.
    const s = data.series.trend_qdm;
    const step = Math.max(1, Math.floor(s.dates.length / 240));
    const idx = s.dates.map((_, i) => i).filter((i) => i % step === 0);
    const labels = idx.map((i) => s.dates[i].slice(2));
    const [c1, c2] = seriesColors();
    const series = [
      { name: "原始预报", color: c1, points: idx.map((i) => ({ x: 0, y: s.raw_series[i] })) },
      { name: "QDM 校正", color: c2, points: idx.map((i) => ({ x: 0, y: s.corrected_series[i] })) },
    ];
    Charts.lineChart($("seriesChart"), {
      series, xLabels: labels, height: 320,
      xTitle: "日期（抽样显示）", yTitle: "降水量 mm", yFmt: fmt2
    });
  }

  // --------------------------------------------------------------- CV
  async function runCv() {
    const data = await post("/api/cv", collectConfig());
    renderCv(data);
  }
  function renderCv(data) {
    $("cvPlan").innerHTML = `<p class="hint">${data.plan.map((p) => p).join("<br>")}</p>
      <div class="legend"><span class="item"><span class="dot" style="background:var(--series-1)"></span>训练</span>
      <span class="item"><span class="dot" style="background:var(--series-2)"></span>选择</span>
      <span class="item"><span class="dot" style="background:var(--series-3)"></span>评估</span></div>`;
    const rec = Object.entries(data.recommendation)
      .map(([k, v]) => `<span class="pill">${k}: ${v[0]} + ${v[1]}</span>`).join(" ");
    let html = `<p><b>跨折投票推荐：</b>${rec}</p>`;
    data.folds.forEach((f) => {
      const total = f.train_years.length + f.val_years.length + f.test_years.length;
      const seg = (n) => (n / total * 100).toFixed(2) + "%";
      let seasonHtml = "";
      Object.entries(f.seasons).forEach(([lab, r]) => {
        if (r.test_metrics === null) {
          seasonHtml += `<div class="cv-season"><b>${lab}</b> · <span class="pill">该季角色块样本不足，已跳过</span></div>`;
          return;
        }
        const wf = r.test_metrics.wet_frequency;
        const mae = r.test_metrics.amount_mae;
        const cands = r.candidates.map((c) =>
          `<span class="pill ${c.occ_method === r.selected[0] && c.amount_method === r.selected[1] ? "sel" : ""}">${c.occ_method}/${c.amount_method} score=${fmt3(c.score)}</span>`).join(" ");
        seasonHtml += `<div class="cv-season"><b>${lab}</b> · 选中 ${r.selected.join(" + ")}
          <div style="margin:6px 0">${cands}</div>
          <div class="kv">
            <div>评估块湿日比例 原始/校正/观测</div><b>${fmt3(wf.raw)} / ${fmt3(wf.corrected)} / ${fmt3(wf.obs)}</b>
            <div>湿日量 MAE 原始/校正</div><b>${fmt2(mae.raw)} / ${fmt2(mae.corrected)}</b>
            <div>样本数 训练/选择/评估</div><b>${r.sizes.join(" / ")}</b>
          </div></div>`;
      });
      html += `<div class="cv-fold"><div class="fold-head">
          <b>折 ${f.fold}</b><span class="mono">训 ${f.train_years[0]}–${f.train_years.at(-1)} · 选 ${f.val_years[0]}–${f.val_years.at(-1)} · 评 ${f.test_years[0]}–${f.test_years.at(-1)}</span>
        </div>
        <div class="seg-bar">
          <div class="seg-train" style="width:${seg(f.train_years.length)}"></div>
          <div class="seg-val" style="width:${seg(f.val_years.length)}"></div>
          <div class="seg-test" style="width:${seg(f.test_years.length)}"></div>
        </div>${seasonHtml}</div>`;
    });
    $("cvDetail").innerHTML = html;
    const style = document.createElement("style");
    style.textContent = ".pill.sel{outline:2px solid var(--series-3);font-weight:700}";
    $("cvDetail").appendChild(style);
  }

  // --------------------------------------------------------------- calibrate
  async function runCalibrate() {
    const body = collectConfig();
    body.use_recommendation = true;
    const data = await post("/api/calibrate", body);
    await refreshSummary();
    $("explainCal").insertAdjacentHTML("beforeend",
      `<option value="${data.calibration_id}">${data.calibration_id}</option>`);
    await runCompare();
  }

  async function runExplain() {
    const calId = $("explainCal").value;
    const row = parseInt($("explainRow").value, 10);
    if (!calId) return showError("请先标定并选择一个 calibration_id");
    const e = await api(`/api/explain?calibration_id=${encodeURIComponent(calId)}&row_index=${row}`);
    const o = e.occurrence, a = e.amount;
    const binding = e.binding;
    $("explainOutput").innerHTML = `
      <div class="kv" style="margin-bottom:8px">
        <div>日期</div><b>${e.date}</div>
        <div>季节</div><b>${e.season}</b>
        <div>原始值</div><b>${fmt2(e.raw_value)} mm</b>
        <div>校正值</div><b>${fmt2(e.corrected_value)} mm</b>
        <div>观测修订绑定</div><b class="${binding.status === "current" ? "status-current" : "status-stale"}">${binding.status} · ${binding.message}</b>
      </div>
      <div class="stage"><b>阶段一 · 发生与否（${o.decision_cutoff ? "" : ""}cutoff=${fmt3(o.decision_cutoff)}）</b>
        <div class="kv">
          <div>经验 CDF F(x) / F(x−)</div><b>${fmt3(o.empirical_cdf)} / ${fmt3(o.empirical_cdf_left)}</b>
          <div>均匀抽样 u</div><b>${fmt3(o.uniform_draw)}</b>
          <div>随机化分位</div><b>${fmt3(o.randomized_quantile)}</b>
          <div>判定为湿日</div><b>${o.is_wet ? "是" : "否（置零）"}</b>
        </div>
      </div>
      ${a ? `<div class="stage amount"><b>阶段二 · 湿日量级</b>
        <div class="kv">
          <div>非超越概率 τ</div><b>${fmt3(a.non_exceedance_tau)}</b>
          <div>观测分位 Q_obs(τ)</div><b>${fmt2(a.obs_quantile_q_obs)}</b>
          <div>历史预报分位 Q_hind(τ)</div><b>${fmt2(a.hind_quantile_q_hind)}</b>
          <div>保留的变化量 Δ</div><b>${fmt2(a.preserved_delta)}</b>
          <div>是否超训练最大值</div><b>${a.beyond_training_max ? "是（走尾部处理）" : "否"}</b>
        </div>
      </div>` : `<div class="stage amount"><b>阶段二 · 量级</b><div class="kv"><div>该日发生阶段判定为干日</div><b>不进入量级映射，校正值为 0</b></div></div>`}`;
  }

  // --------------------------------------------------------------- calibration list / export
  async function renderCalibrationList(items) {
    const sel = $("explainCal");
    const current = sel.value;
    sel.innerHTML = "";
    if (!items || !items.length) {
      $("calibrationList").innerHTML = `<p class="hint">尚无标定。配置后点击“标定并校正未来”。</p>`;
      $("exportBtn").disabled = true;
      return;
    }
    $("exportBtn").disabled = false;
    $("calibrationList").innerHTML = items.map((c) => {
      const methods = Object.entries(c.selected_methods)
        .map(([k, v]) => `<span class="pill">${k}: ${v[0]}+${v[1]}</span>`).join(" ");
      const b = c.binding;
      const status = b ? `<span class="${b.status === "current" ? "status-current" : "status-stale"}">${b.status === "current" ? "绑定当前观测" : "观测已修订·仍绑定原标定"}</span>` : "";
      return `<div class="cal-item"><div class="cal-head">
          <label style="flex-direction:row;align-items:center;gap:8px">
            <input type="radio" name="calpick" value="${c.calibration_id}" ${items.length === 1 ? "checked" : ""}>
            <span class="mono">${c.calibration_id}</span></label>
          ${status}
        </div>
        <div class="kv" style="margin-top:8px">
          <div>标定指纹</div><b class="mono">${c.fingerprint}</b>
          <div>观测修订 / 指纹</div><b class="mono">${c.obs_revision} · ${(c.obs_fingerprint || "").slice(0, 12)}…</b>
          <div>方法</div><b>${methods}</b>
        </div></div>`;
    }).join("");
    items.forEach((c) => sel.insertAdjacentHTML("beforeend",
      `<option value="${c.calibration_id}">${c.calibration_id}</option>`));
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  }

  function pickedCalibration() {
    const radio = document.querySelector('input[name="calpick"]:checked');
    if (radio) return radio.value;
    return $("explainCal").value;
  }

  function exportCsv() {
    const id = pickedCalibration();
    if (!id) return showError("请选择一个标定");
    window.location = `/api/export?calibration_id=${encodeURIComponent(id)}`;
  }

  // --------------------------------------------------------------- revision
  async function revise() {
    const data = await post("/api/data/revise", { multiplier: 1.1, addend: 0 });
    await refreshSummary();
    showError(`观测已修订为 ${data.obs_revision}；旧产品仍绑定原标定（见版本区状态）。`);
  }

  // --------------------------------------------------------------- wiring
  function wire() {
    $("themeToggle").addEventListener("click", () => {
      const body = document.body;
      const dark = body.dataset.theme === "dark";
      body.dataset.theme = dark ? "light" : "dark";
      $("themeToggle").textContent = dark ? "切换深色" : "切换浅色";
      if (lastCompare) { renderWetFreq(lastCompare); renderQuantiles(lastCompare);
        renderTail(lastCompare); renderSeries(lastCompare); }
    });
    $("loadSynthetic").addEventListener("click", async () => {
      try { await api("/api/data/synthetic", { method: "POST" }); await refreshSummary(); }
      catch (e) { showError(e.message); }
    });
    $("historyFile").addEventListener("change", (e) => { historyFile = e.target.files[0]; checkImport(); });
    $("futureFile").addEventListener("change", (e) => { futureFile = e.target.files[0]; checkImport(); });
    function checkImport() { $("importBtn").disabled = !(historyFile && futureFile); }
    $("importBtn").addEventListener("click", async () => {
      const fd = new FormData();
      fd.append("history", historyFile); fd.append("future", futureFile);
      try {
        const res = await fetch("/api/data/import", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        await refreshSummary();
      } catch (e) { showError(e.message); }
    });
    $("runCv").addEventListener("click", () => runCv().catch((e) => showError(e.message)));
    $("runCalibrate").addEventListener("click", () => runCalibrate().catch((e) => showError(e.message)));
    $("runCompare").addEventListener("click", () => runCompare().catch((e) => showError(e.message)));
    $("explainBtn").addEventListener("click", () => runExplain().catch((e) => showError(e.message)));
    $("exportBtn").addEventListener("click", exportCsv);
    $("reviseBtn").addEventListener("click", () => revise().catch((e) => showError(e.message)));
    $("seasonMode").addEventListener("change", () => {
      const annual = $("seasonMode").value === "annual";
      $("seasonSelect").disabled = annual;
      if (annual) $("seasonSelect").value = "JJA";
    });
  }

  wire();
  refreshSummary().catch(() => {});
})();
