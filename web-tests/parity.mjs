// Node parity test: run the TF.js graph models with the same inference code the
// browser uses (web/inference.js) and compare with the Python outputs recorded
// by scripts/export_web.py. Run with: npm run test:parity
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import { PNG } from "pngjs";
import { predict, preprocess } from "../web/inference.js";

const require = createRequire(import.meta.url);
const TOL = { logit: 1e-3, prob: 1e-4, cam: 1e-3 };

const ref = JSON.parse(readFileSync("web-tests/reference.json", "utf8"));

function loadHandler(dir) {
  return {
    async load() {
      const json = JSON.parse(readFileSync(path.join(dir, "model.json"), "utf8"));
      const specs = [];
      const bufs = [];
      for (const group of json.weightsManifest) {
        specs.push(...group.weights);
        for (const p of group.paths) bufs.push(readFileSync(path.join(dir, p)));
      }
      const all = Buffer.concat(bufs);
      return {
        modelTopology: json.modelTopology,
        format: json.format,
        convertedBy: json.convertedBy,
        signature: json.signature,
        userDefinedMetadata: json.userDefinedMetadata,
        weightSpecs: specs,
        weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
      };
    },
  };
}

function readPng(file) {
  const png = PNG.sync.read(readFileSync(file));
  const rgb = new Int32Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) {
    rgb[j] = png.data[i];
    rgb[j + 1] = png.data[i + 1];
    rgb[j + 2] = png.data[i + 2];
  }
  return tf.tensor3d(rgb, [png.height, png.width, 3], "int32");
}

function maxAbs(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

let failures = 0;
const worst = { logit: 0, prob: 0, cam: 0 };

async function runBackend(backend) {
  if (backend === "wasm") {
    const dist = path.dirname(require.resolve("@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm"));
    setWasmPaths(dist + path.sep);
  }
  await tf.setBackend(backend);
  await tf.ready();
  for (const name of Object.keys(ref.models)) {
    const meta = JSON.parse(readFileSync(`web/models/${name}/meta.json`, "utf8"));
    const model = await tf.loadGraphModel(loadHandler(`web/models/${name}`));
    const cases = [
      ...ref.samples.map((s, i) => ({ file: `web/samples/${s.file}`, expect: ref.models[name].samples[i] })),
      ...ref.uploads
        .map((u, i) => ({ u, expect: ref.models[name].uploads[i] }))
        .filter(({ u }) => u.lossless)
        .map(({ u, expect }) => ({ file: `web-tests/fixtures/${u.file}`, expect })),
    ];
    for (const c of cases) {
      const pixels = readPng(c.file);
      const x = preprocess(tf, pixels);
      const r = await predict(tf, model, meta, x);
      pixels.dispose();
      x.dispose();
      const d = {
        logit: Math.abs(r.logit - c.expect.logit),
        prob: Math.abs(r.prob - c.expect.prob),
        cam: maxAbs(r.camTumor, c.expect.cam.flat().map((v) => Math.max(0, v))),
      };
      for (const k of Object.keys(d)) worst[k] = Math.max(worst[k], d[k]);
      const ok = d.logit <= TOL.logit && d.prob <= TOL.prob && d.cam <= TOL.cam;
      if (!ok) {
        failures++;
        console.log(`FAIL ${backend} ${name} ${c.file}`, d);
      }
    }
    console.log(`${backend} ${name}: ${cases.length} images checked`);
  }
}

for (const b of ["cpu", "wasm"]) await runBackend(b);
console.log("max abs differences", worst, "tolerances", TOL);
if (failures) {
  console.error(`${failures} parity failures`);
  process.exit(1);
}
console.log("parity OK");
