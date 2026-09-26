// Shared inference code for the browser demo and the Node parity test.
// Every function takes the TF.js namespace as its first argument so the same
// code runs with the browser bundle and with @tensorflow/tfjs in Node.

export const INPUT_SIZE = 224;

/**
 * RGB pixels (H, W, 3) -> model input (1, 224, 224, 3) float32 in [0, 255].
 * Bilinear with half-pixel centers and no antialiasing, which matches
 * tf.image.resize(..., "bilinear") used on the Python side.
 */
export function preprocess(tf, pixels) {
  return tf.tidy(() => {
    const x = pixels.toFloat();
    const [h, w] = x.shape;
    const resized =
      h === INPUT_SIZE && w === INPUT_SIZE
        ? x
        : tf.image.resizeBilinear(x, [INPUT_SIZE, INPUT_SIZE], false, true);
    return resized.expandDims(0);
  });
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/**
 * Run a binary model on a preprocessed input.
 * Returns the raw logit, the temperature-calibrated tumor probability, the
 * predicted label, and the Grad-CAM map for the predicted class
 * (ReLU of the signed map, normalised to [0, 1]) as a flat array.
 */
export async function predict(tf, model, meta, x) {
  // The converted signature exposes the outputs under their SavedModel names.
  const outs = model.execute(x, ["logit", "cam"]);
  const [logit, cam] = outs;
  const z = (await logit.data())[0];
  const [, h, w] = cam.shape;
  const raw = await cam.data();
  outs.forEach((t) => t.dispose());

  const prob = sigmoid(z / meta.temperature);
  const label = prob >= meta.threshold ? "tumor" : "healthy";
  // Grad-CAM for the predicted class: the healthy logit is -z, so flip the sign.
  const sign = label === "tumor" ? 1 : -1;
  const camPred = new Float32Array(raw.length);
  const camTumor = new Float32Array(raw.length);
  let max = 0;
  for (let i = 0; i < raw.length; i++) {
    camTumor[i] = Math.max(0, raw[i]);
    camPred[i] = Math.max(0, sign * raw[i]);
    if (camPred[i] > max) max = camPred[i];
  }
  if (max > 0) for (let i = 0; i < camPred.length; i++) camPred[i] /= max;
  return { logit: z, prob, label, camTumor, camPred, camShape: [h, w] };
}
