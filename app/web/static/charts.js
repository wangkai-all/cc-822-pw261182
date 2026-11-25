/* Minimal dependency-free SVG charting with crosshair + tooltip.
   Reads CSS variables for theme tokens so light/dark swap automatically. */
(function () {
  const css = (name) => getComputedStyle(document.querySelector(".viz-root"))
    .getPropertyValue(name).trim();
  const SERIES = () => [css("--series-1"), css("--series-2"), css("--series-3")];

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceTicks(min, max, count = 5) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(+v.toFixed(6));
    return { ticks, min, max };
  }

  const Tooltip = {
    node: null,
    init() { if (!this.node) this.node = document.getElementById("tooltip"); },
    show(x, y, html) {
      this.init(); this.node.innerHTML = html; this.node.hidden = false;
      const r = this.node.getBoundingClientRect();
      const left = Math.min(x + 14, window.innerWidth - r.width - 10);
      const top = Math.min(y + 14, window.innerHeight - r.height - 10);
      this.node.style.left = left + "px";
      this.node.style.top = top + "px";
    },
    hide() { this.init(); this.node.hidden = true; }
  };

  function baseFrame(container, height) {
    container.innerHTML = "";
    const width = Math.max(320, container.clientWidth);
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" }, container);
    const margin = { top: 18, right: 22, bottom: 42, left: 56 };
    const plot = {
      x: margin.left, y: margin.top,
      w: width - margin.left - margin.right,
      h: height - margin.top - margin.bottom
    };
    return { svg, plot, width, height };
  }

  function drawAxes(svg, plot, xTicks, yTicks, xFmt, yFmt) {
    const g = el("g", { class: "axis" }, svg);
    // horizontal grid + y labels
    yTicks.ticks.forEach((t) => {
      const y = plot.y + plot.h - (t - yTicks.min) / (yTicks.max - yTicks.min) * plot.h;
      el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline" }, g);
      el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end", text: yFmt(t) }, g);
    });
    // x labels
    xTicks.labels.forEach((lab, i) => {
      const x = plot.x + (xTicks.positions[i]) * plot.w;
      el("text", { x, y: plot.y + plot.h + 18, "text-anchor": "middle", text: lab }, g);
    });
    el("line", { x1: plot.x, x2: plot.x, y1: plot.y, y2: plot.y + plot.h, class: "axis-line" }, g);
    el("line", { x1: plot.x, x2: plot.x + plot.w, y1: plot.y + plot.h, y2: plot.y + plot.h, class: "axis-line" }, g);
  }

  // Generic multi-series line chart. series: [{name,color,points:[{x,y,meta}]}]
  function lineChart(container, opts) {
    const { series, xLabels, height = 300, xTitle, yTitle, yFmt = (d) => d.toFixed(2) } = opts;
    const { svg, plot } = baseFrame(container, height);
    const allY = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => isFinite(v));
    const yMin = Math.min(0, ...allY), yMax = Math.max(...allY);
    const yTicks = niceTicks(yMin, yMax);
    const n = xLabels.length;
    const xPositions = xLabels.map((_, i) => (n === 1 ? 0.5 : i / (n - 1)));
    drawAxes(svg, plot, { labels: xLabels, positions: xPositions }, yTicks,
      (d) => d, yFmt);
    const sx = (i) => plot.x + xPositions[i] * plot.w;
    const sy = (v) => plot.y + plot.h - (v - yTicks.min) / (yTicks.max - yTicks.min) * plot.h;

    series.forEach((s) => {
      const d = s.points.map((p, i) =>
        `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
      el("path", { d, class: "series-line", stroke: s.color }, svg);
      // Direct label at the last point (relief for contrast + identity).
      const last = s.points[s.points.length - 1];
      el("text", {
        x: plot.x + plot.w + 4, y: sy(last.y) + 4, class: "direct-label",
        fill: s.color, text: s.name
      }, svg);
    });

    // Crosshair + hover.
    const cross = el("line", { y1: plot.y, y2: plot.y + plot.h, stroke: css("--baseline"),
      "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const dots = series.map((s) => el("circle", { r: 4, fill: s.color, stroke: css("--surface"),
      "stroke-width": 2, visibility: "hidden" }, svg));
    const overlay = el("rect", { x: plot.x, y: plot.y, width: plot.w, height: plot.h,
      fill: "transparent" }, svg);
    overlay.addEventListener("mousemove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const rel = (ev.clientX - rect.left) - plot.x;
      const i = Math.round((rel / plot.w) * (n - 1));
      if (i < 0 || i >= n) return;
      const x = sx(i);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x);
      cross.setAttribute("visibility", "visible");
      let html = `<div class="tt-title">${xLabels[i]}</div>`;
      series.forEach((s, k) => {
        const p = s.points[i];
        dots[k].setAttribute("cx", x); dots[k].setAttribute("cy", sy(p.y));
        dots[k].setAttribute("visibility", "visible");
        const val = p.meta && p.meta.label ? p.meta.label : yFmt(p.y);
        html += `<div class="tt-row"><span><span style="color:${s.color}">●</span> ${s.name}</span><b>${val}</b></div>`;
      });
      Tooltip.show(ev.clientX, ev.clientY, html);
    });
    overlay.addEventListener("mouseleave", () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      Tooltip.hide();
    });

    if (xTitle) el("text", { x: plot.x + plot.w / 2, y: height - 6, "text-anchor": "middle",
      fill: css("--muted"), text: xTitle }, svg);
    if (yTitle) el("text", { x: 14, y: plot.y + plot.h / 2, "text-anchor": "middle",
      fill: css("--muted"), transform: `rotate(-90 14 ${plot.y + plot.h / 2})`, text: yTitle }, svg);
  }

  // Grouped bar chart for categorical comparison (e.g. wet frequency).
  function barChart(container, opts) {
    const { groups, height = 260, yFmt = (d) => d.toFixed(3) } = opts;
    const { svg, plot } = baseFrame(container, height);
    const colors = SERIES();
    const allVals = groups.flatMap((g) => g.values.map((v) => v.value));
    const yTicks = niceTicks(0, Math.max(...allVals) * 1.1);
    const sy = (v) => plot.y + plot.h - (v - yTicks.min) / (yTicks.max - yTicks.min) * plot.h;
    yTicks.ticks.forEach((t) => {
      const y = sy(t);
      el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline" }, svg);
      el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end", fill: css("--muted"), text: yFmt(t) }, svg);
    });
    const gw = plot.w / groups.length;
    const nser = groups[0].values.length;
    const bw = (gw * 0.62) / nser;
    groups.forEach((g, gi) => {
      const gx = plot.x + gi * gw + gw * 0.19;
      g.values.forEach((v, si) => {
        const x = gx + si * bw;
        const y = sy(v.value);
        const rect = el("rect", { x, y, width: bw, height: plot.y + plot.h - y,
          rx: 4, fill: colors[si % colors.length] }, svg);
        rect.addEventListener("mousemove", (ev) => Tooltip.show(ev.clientX, ev.clientY,
          `<div class="tt-title">${g.label}</div>
           <div class="tt-row"><span><span style="color:${colors[si % colors.length]}">●</span> ${v.name}</span><b>${yFmt(v.value)}</b></div>`));
        rect.addEventListener("mouseleave", () => Tooltip.hide());
      });
      el("text", { x: gx + gw * 0.31, y: plot.y + plot.h + 18, "text-anchor": "middle",
        fill: css("--muted"), text: g.label }, svg);
    });
  }

  function legend(container, items) {
    const colors = SERIES();
    const wrap = document.createElement("div");
    wrap.className = "legend";
    items.forEach((it, i) => {
      const span = document.createElement("span");
      span.className = "item";
      span.innerHTML = `<span class="dot" style="background:${it.color || colors[i % colors.length]}"></span>${it.name}`;
      wrap.appendChild(span);
    });
    container.appendChild(wrap);
  }

  window.Charts = { lineChart, barChart, legend, Tooltip, SERIES };
})();
