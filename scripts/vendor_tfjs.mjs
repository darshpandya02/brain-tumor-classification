// Copy the TF.js runtime and WASM backend into web/vendor so the demo loads
// nothing from third-party servers.
import { copyFileSync, mkdirSync } from "node:fs";

const out = "web/vendor";
mkdirSync(out, { recursive: true });
const files = [
  ["node_modules/@tensorflow/tfjs/dist/tf.min.js", "tf.min.js"],
  ["node_modules/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.min.js", "tf-backend-wasm.min.js"],
  ["node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm", "tfjs-backend-wasm.wasm"],
  ["node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm", "tfjs-backend-wasm-simd.wasm"],
  ["node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm", "tfjs-backend-wasm-threaded-simd.wasm"],
];
for (const [src, dst] of files) copyFileSync(src, `${out}/${dst}`);
console.log(`copied ${files.length} files to ${out}`);
