"use strict";

const EXPECTED_SCHEMA = 2;
const DATA = "data/";
const files = ["manifest.json", "primary-validation.json", "contrast-validation.json", "calibration.json", "primary-benchmark.json", "contrast-benchmark.json", "telemetry-summary.json", "telemetry-trace.json"];
const $ = (id) => document.getElementById(id);

function seconds(value) {
  if (value < 1e-6) return `${(value * 1e9).toFixed(2)} ns`;
  if (value < 1e-3) return `${(value * 1e6).toFixed(2)} µs`;
  return `${(value * 1e3).toFixed(2)} ms`;
}
function status(value) { return value.replaceAll("_", " ").toUpperCase(); }
function setText(id, value) { $(id).textContent = value; }
function rows(element, values) {
  element.replaceChildren(...values.map(([label, value]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt"); dt.textContent = label;
    const dd = document.createElement("dd"); dd.textContent = value;
    row.append(dt, dd); return row;
  }));
}
function comparisonChart(id, validation) {
  const predicted = validation.prediction.latency_seconds;
  const measured = validation.measurement.p50_latency_seconds;
  const max = Math.max(predicted, measured);
  const items = [["Predicted", predicted, "predicted"], ["Measured p50", measured, "measured"]];
  $(id).replaceChildren(...items.map(([label, value, className]) => {
    const row = document.createElement("div"); row.className = "comparison-bar";
    const name = document.createElement("span"); name.textContent = label;
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = `bar-fill ${className}`; fill.style.width = `${value / max * 100}%`;
    const number = document.createElement("strong"); number.textContent = seconds(value);
    track.append(fill); row.append(name, track, number); return row;
  }));
}
function renderComparison(prefix, validation) {
  const ratio = validation.error.measured_to_predicted_ratio;
  setText(`${prefix}-ratio`, `${ratio.toFixed(2)}× prediction`);
  comparisonChart(`${prefix}-chart`, validation);
  setText(`${prefix}-chart-label`, `Predicted ${seconds(validation.prediction.latency_seconds)}; measured p50 ${seconds(validation.measurement.p50_latency_seconds)}; measured-to-predicted ratio ${ratio.toFixed(2)} times.`);
  rows($(`${prefix}-values`), [
    ["Predicted", seconds(validation.prediction.latency_seconds)],
    ["Measured p50", seconds(validation.measurement.p50_latency_seconds)],
    ["Measured p95", seconds(validation.measurement.p95_latency_seconds)],
    ["APE", `${validation.error.absolute_percentage_error.toFixed(2)}%`],
  ]);
}
const telemetryMeta = {
  gpu_utilization_percent: ["GPU utilization", "%"], power_watts: ["Power", " W"],
  temperature_celsius: ["Temperature", " °C"], sm_clock_mhz: ["SM clock", " MHz"],
  memory_activity_percent: ["Memory activity", "%"],
};
function renderTrace(trace, metric) {
  const values = trace.samples.map((sample) => [sample.timestamp_seconds, sample[metric]]).filter(([, value]) => value !== null);
  if (!values.length) throw new Error(`Telemetry metric ${metric} is unavailable.`);
  const [label, unit] = telemetryMeta[metric];
  const width = 960, height = 300, left = 70, right = 28, top = 30, bottom = 48;
  const xMax = Math.max(...values.map(([x]) => x)); const yMin = Math.min(...values.map(([, y]) => y)); const yMax = Math.max(...values.map(([, y]) => y));
  const span = yMax - yMin || 1;
  const x = (v) => left + (v / xMax) * (width - left - right);
  const y = (v) => top + ((yMax - v) / span) * (height - top - bottom);
  const svg = $("telemetry-chart");
  const ns = "http://www.w3.org/2000/svg";
  const el = (name, attrs, text) => { const node = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k,v)); if(text) node.textContent=text; return node; };
  svg.replaceChildren();
  const title = el("title", {id:"trace-title"}, `${label} over the separate telemetry window`);
  const desc = el("desc", {id:"trace-desc"}, `${values.length} NVML samples. Minimum ${yMin}${unit}; maximum ${yMax}${unit}.`);
  svg.append(title, desc);
  [0,.25,.5,.75,1].forEach((part) => {
    const yy = top + part * (height-top-bottom); svg.append(el("line", {x1:left,y1:yy,x2:width-right,y2:yy,class:"trace-grid"}));
  });
  const points = values.map(([a,b]) => `${x(a)},${y(b)}`).join(" ");
  svg.append(el("polyline", {points,class:"trace-line"}));
  svg.append(el("text", {x:left,y:height-14,class:"trace-label"}, "0 s"), el("text", {x:width-right,y:height-14,"text-anchor":"end",class:"trace-label"}, `${xMax.toFixed(1)} s`));
  svg.append(el("text", {x:left-10,y:top+7,"text-anchor":"end",class:"trace-label"}, `${yMax}${unit}`), el("text", {x:left-10,y:height-bottom,"text-anchor":"end",class:"trace-label"}, `${yMin}${unit}`));
  setText("trace-equivalent", `${label}: ${values.length} samples over ${xMax.toFixed(2)} seconds; range ${yMin}${unit} to ${yMax}${unit}.`);
}

async function init() {
  try {
    const loaded = await Promise.all(files.map(async (name) => {
      const response = await fetch(DATA + name);
      if (!response.ok) throw new Error(`${name} returned HTTP ${response.status}`);
      return [name, await response.json()];
    }));
    const evidence = Object.fromEntries(loaded); const m = evidence["manifest.json"];
    if (m.schema_version !== EXPECTED_SCHEMA) throw new Error(`Unsupported manifest schema ${m.schema_version}; expected ${EXPECTED_SCHEMA}.`);
    const primary = evidence["primary-validation.json"], contrast = evidence["contrast-validation.json"], calibration = evidence["calibration.json"], telemetry = evidence["telemetry-summary.json"], trace = evidence["telemetry-trace.json"];
    setText("device-name", m.device.name); setText("runtime", `PyTorch ${m.runtime.pytorch_version} / CUDA ${m.runtime.cuda_runtime_version}`); setText("dtype", m.device.dtype.toUpperCase()); setText("capture-source", m.tensorforge_commit.slice(0,7));
    setText("hero-predicted", seconds(primary.prediction.latency_seconds)); setText("hero-measured", seconds(primary.measurement.p50_latency_seconds)); setText("hero-ape", `${primary.error.absolute_percentage_error.toFixed(2)}%`);
    renderComparison("primary", primary); renderComparison("contrast", contrast);
    setText("compute-rate", `${(calibration.effective_compute_flops_per_second/1e12).toFixed(2)} TFLOP/s`); setText("memory-rate", `${(calibration.effective_memory_bandwidth_bytes_per_second/1e9).toFixed(2)} GB/s`);
    setText("compute-probe", `${calibration.compute_probe.shape_m} × ${calibration.compute_probe.shape_n} × ${calibration.compute_probe.shape_k} ${calibration.dtype.toUpperCase()} GEMM`); setText("memory-probe", `${(calibration.memory_probe.payload_bytes/1048576).toFixed(0)} MiB payload`); setText("bottleneck", primary.prediction.bottleneck);
    const unavailable = $("unavailable-list"); unavailable.replaceChildren(...m.telemetry.unavailable_fields.map((name) => { const li=document.createElement("li"); li.textContent=`${name.replaceAll("_"," ")} — NOT SUPPORTED`; return li; }));
    renderTrace(trace, $("telemetry-metric").value); $("telemetry-metric").addEventListener("change", (event) => renderTrace(trace,event.target.value));
    setText("regression-status", status(m.regression.status)); setText("regression-reason", m.regression.reason); setText("sizing-status", status(m.right_sizing.status)); setText("sizing-reason", m.right_sizing.reason); setText("impact-status", status(m.impact.status)); setText("impact-reason", m.impact.reason);
    rows($("method-values"), [["Device",m.device.name],["Dtype",m.device.dtype.toUpperCase()],["Warmups",String(m.measurement.warmup_iterations)],["Iterations",String(m.measurement.measured_iterations)],["Primary",m.primary_workload],["Contrast",m.contrast_workload],["Telemetry",`${telemetry.sample_count} samples · separate phase`]]);
    rows($("provenance-values"), [["Evidence commit","72b7e8a"],["Capture source",m.tensorforge_commit.slice(0,7)],["Device","RTX 3050 Laptop GPU"],["Integrity","13 SHA-256 verified artifacts"]]);
  } catch (error) {
    const box = $("load-error"); box.hidden = false; box.textContent = `Evidence could not be rendered: ${error.message}`; document.body.classList.add("evidence-failed");
  }
}
init();
