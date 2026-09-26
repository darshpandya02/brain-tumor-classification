import { INPUT_SIZE, predict, preprocess } from "./inference.js";

const tf = window.tf;
const $ = (id) => document.getElementById(id);

const state = {
  backend: null,
  models: {}, // name -> { meta, model }
  current: null,
  samples: [],
  lastInput: null, // { kind, index, label, name }
  lastResult: null,
  busy: false,
};

// Exposed for automated end-to-end checks.
window.__demo = { state, ready: false, runs: 0 };

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function initBackend() {
  try {
    tf.wasm.setWasmPaths(new URL("vendor/", window.location.href).href);
    if (await tf.setBackend("wasm")) {
      await tf.ready();
      return "wasm";
    }
  } catch (e) {
    console.warn("WASM backend unavailable, falling back to CPU", e);
  }
  await tf.setBackend("cpu");
  await tf.ready();
  return "cpu";
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function loadModel(name) {
  if (state.models[name]?.model) return state.models[name];
  setStatus("Loading model...");
  const meta = state.models[name]?.meta ?? (await fetchJSON(`models/${name}/meta.json`));
  const model = await tf.loadGraphModel(`models/${name}/model.json`);
  // Warm-up run so the first real prediction is fast.
  tf.tidy(() => {
    model.execute(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]), model.outputNodes);
  });
  state.models[name] = { meta, model };
  return state.models[name];
}

function pct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

function renderMetrics(metas) {
  const rows = metas
    .map((m) => {
      const t = m.test_metrics;
      const ci = m.test_ci95;
      const cell = (k) => `${pct(t[k])}<span class="ci">${pct(ci[k][0])} to ${pct(ci[k][1])}</span>`;
      return `<tr><th scope="row">${m.title}</th><td>${cell("accuracy")}</td><td>${cell("sensitivity")}</td>
        <td>${cell("specificity")}</td><td>${t.roc_auc.toFixed(3)}<span class="ci">${ci.roc_auc[0].toFixed(3)} to ${ci.roc_auc[1].toFixed(3)}</span></td>
        <td>${t.ece.toFixed(3)}</td>
        <td class="cm">TN ${t.confusion_matrix[0][0]}, FP ${t.confusion_matrix[0][1]}<br>FN ${t.confusion_matrix[1][0]}, TP ${t.confusion_matrix[1][1]}</td></tr>`;
    })
    .join("");
  const n = metas[0].test_metrics.n;
  const cm = metas[0].test_metrics.confusion_matrix;
  $("metrics").innerHTML = `<table class="metrics-table">
    <caption>Test set: ${n} images (${cm[0][0] + cm[0][1]} healthy, ${cm[1][0] + cm[1][1]} tumor), threshold 0.5</caption>
    <thead><tr><th>Model</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>ECE</th><th>Confusion matrix</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderSamples() {
  const box = $("samples");
  box.innerHTML = "";
  state.samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "sample";
    b.dataset.index = String(i);
    b.dataset.label = s.label;
    b.title = `True label: ${s.label}${s.tumor_type ? ` (${s.tumor_type})` : ""}`;
    b.innerHTML = `<img src="samples/${s.file}" alt="Sample MRI ${i + 1}, true label ${s.label}" width="96" height="96">
      <span class="badge ${s.label}">${s.label}${s.tumor_type ? `: ${s.tumor_type}` : ""}</span>`;
    b.addEventListener("click", () => runSample(i));
    box.appendChild(b);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode the image."));
    img.src = src;
  });
}

// Perceptually ordered colormap (approximation of "turbo").
function turbo(t) {
  const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
  const c = (v) => Math.max(0, Math.min(255, Math.round(v)));
  return [c(r), c(g), c(b)];
}

async function drawCam(result) {
  const [h, w] = result.camShape;
  const up = tf.tidy(() =>
    tf.image
      .resizeBilinear(tf.tensor3d(result.camPred, [h, w, 1]), [INPUT_SIZE, INPUT_SIZE], false, true)
      .clipByValue(0, 1)
  );
  const v = await up.data();
  up.dispose();
  const ctx = $("cam-canvas").getContext("2d");
  const img = ctx.createImageData(INPUT_SIZE, INPUT_SIZE);
  for (let i = 0; i < v.length; i++) {
    const [r, g, b] = turbo(v[i]);
    img.data[4 * i] = r;
    img.data[4 * i + 1] = g;
    img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = Math.round(255 * Math.min(1, v[i] * 1.4));
  }
  ctx.putImageData(img, 0, 0);
  $("cam-canvas").dataset.rendered = "true";
  $("cam-canvas").dataset.maxValue = String(Math.max(...result.camPred));
}

async function classify(source, info) {
  if (state.busy) return;
  state.busy = true;
  const { meta, model } = state.models[state.current];
  setStatus("Running...");
  const t0 = performance.now();
  let x;
  try {
    x = tf.tidy(() => preprocess(tf, tf.browser.fromPixels(source)));
    const shown = tf.tidy(() => x.squeeze(0).div(255).clipByValue(0, 1));
    await tf.browser.toPixels(shown, $("image-canvas"));
    shown.dispose();
    const result = await predict(tf, model, meta, x);
    result.ms = performance.now() - t0;
    state.lastResult = result;
    state.lastInput = info;
    await drawCam(result);
    renderResult(result, info, meta);
    window.__demo.runs += 1;
    setStatus(`Ready (${state.backend} backend, ${result.ms.toFixed(0)} ms)`);
  } catch (e) {
    console.error(e);
    setStatus(`Error: ${e.message}`, true);
  } finally {
    x?.dispose();
    state.busy = false;
  }
}

function renderResult(r, info, meta) {
  const el = $("result");
  el.classList.remove("empty");
  const truth = info.label
    ? `<p>True label: <strong>${info.label}</strong>${info.tumorType ? ` (${info.tumorType})` : ""}.
       The model is <strong class="${info.label === r.label ? "ok" : "wrong"}">${info.label === r.label ? "correct" : "wrong"}</strong> on this image.</p>`
    : `<p>Uploaded image: no true label. This result is not a diagnosis.</p>`;
  el.innerHTML = `
    <p class="prediction ${r.label}">Prediction: <strong id="pred-label">${r.label}</strong></p>
    <div class="bar" aria-hidden="true"><div class="bar-fill" style="width:${(100 * r.prob).toFixed(1)}%"></div></div>
    <p>Probability of tumor: <strong id="pred-prob">${pct(r.prob)}</strong>
      <span class="hint">(temperature-scaled, T = ${meta.temperature.toFixed(2)}; threshold ${meta.threshold})</span></p>
    ${truth}
    <p class="hint">Heatmap: Grad-CAM for the predicted class (${r.label}), ${r.camShape[0]}x${r.camShape[1]} feature map
      upsampled to ${INPUT_SIZE}x${INPUT_SIZE}. Model: ${meta.title}.</p>`;
  el.dataset.label = r.label;
  el.dataset.prob = String(r.prob);
  el.dataset.logit = String(r.logit);
  el.dataset.model = meta.name;
  el.dataset.source = info.kind === "sample" ? `sample:${info.index}` : `upload:${info.name}`;
}

async function runSample(i) {
  const s = state.samples[i];
  const img = await loadImage(`samples/${s.file}`);
  document.querySelectorAll(".sample").forEach((b) => b.classList.toggle("selected", b.dataset.index === String(i)));
  await classify(img, { kind: "sample", index: i, label: s.label, tumorType: s.tumor_type });
}

async function runFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file (JPEG or PNG).", true);
    return;
  }
  // Object URL: the file is read locally by the browser, nothing is uploaded.
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    document.querySelectorAll(".sample").forEach((b) => b.classList.remove("selected"));
    await classify(img, { kind: "upload", name: file.name, label: null });
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function rerun() {
  const info = state.lastInput;
  if (!info) return;
  if (info.kind === "sample") await runSample(info.index);
}

async function main() {
  state.backend = await initBackend();
  const index = await fetchJSON("models/index.json");
  const metas = await Promise.all(index.models.map((m) => fetchJSON(`models/${m.name}/meta.json`)));
  metas.forEach((meta) => (state.models[meta.name] = { meta }));
  renderMetrics(metas);

  const select = $("model-select");
  for (const m of index.models) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.title;
    select.appendChild(o);
  }
  // Default to the model with the higher test ROC-AUC.
  const best = metas.reduce((a, b) => (b.test_metrics.roc_auc > a.test_metrics.roc_auc ? b : a));
  select.value = best.name;
  state.current = best.name;
  select.addEventListener("change", async () => {
    select.disabled = true;
    state.current = select.value;
    await loadModel(state.current);
    select.disabled = false;
    setStatus(`Ready (${state.backend} backend)`);
    await rerun();
  });

  state.samples = await fetchJSON("samples/samples.json");
  renderSamples();

  $("file-input").addEventListener("change", (e) => runFile(e.target.files[0]));
  const dz = $("dropzone");
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    runFile(e.dataTransfer.files[0]);
  });
  $("cam-toggle").addEventListener("change", (e) => ($("cam-canvas").style.display = e.target.checked ? "" : "none"));
  $("cam-opacity").addEventListener("input", (e) => ($("cam-canvas").style.opacity = e.target.value));
  $("cam-canvas").style.opacity = $("cam-opacity").value;

  await loadModel(state.current);
  setStatus(`Ready (${state.backend} backend)`);
  window.__demo.ready = true;
  window.__demo.runSample = runSample;
}

main().catch((e) => {
  console.error(e);
  setStatus(`Failed to start: ${e.message}`, true);
});
