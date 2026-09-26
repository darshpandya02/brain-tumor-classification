#!/usr/bin/env bash
# Convert the exported SavedModels to TF.js graph models in web/models/.
#
# tensorflowjs 4.22 (the latest release) declares dependencies that do not
# resolve next to TensorFlow 2.21 (tensorflow-decision-forests, jax), and only
# its SavedModel converter is needed here, so it is installed without deps into
# a separate virtualenv and the two optional imports are neutralised.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV=.venv-tfjs
if [ ! -x "$ENV/bin/tensorflowjs_converter" ]; then
  uv venv -q --python 3.12 "$ENV"
  VIRTUAL_ENV="$ENV" uv pip install -q "tensorflow==2.21.0" tf-keras tensorflow-hub "setuptools<81" packaging six importlib_resources
  VIRTUAL_ENV="$ENV" uv pip install -q --no-deps "tensorflowjs==4.22.0"
  SP=$("$ENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
  mkdir -p "$SP/tensorflow_decision_forests" && : > "$SP/tensorflow_decision_forests/__init__.py"
  "$ENV/bin/python" - "$SP/tensorflowjs/converters/__init__.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
line = "from tensorflowjs.converters.jax_conversion import convert_jax\n"
if line in s:
    s = s.replace(line, "try:\n  " + line + "except ImportError:\n  pass\n")
    open(p, "w").write(s)
PY
fi

for name in ${MODELS:-cnn_binary mobilenet_binary}; do
  rm -f web/models/$name/model.json web/models/$name/group*.bin
  "$ENV/bin/tensorflowjs_converter" --input_format=tf_saved_model --output_format=tfjs_graph_model \
    --signature_name=serving_default "artifacts/$name/saved_model" "web/models/$name" 2>&1 | grep -v -E "^(WARNING|I0000|W0000)" || true
  ls -la "web/models/$name"
done
