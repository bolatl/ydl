#!/usr/bin/env bash
set -eo pipefail

REPO_RAW_BASE="https://raw.githubusercontent.com/cillustrisimo/ydl_2026/main"
ENV_NAME="nn-lab"
PYTHON_VERSION="3.11"
DEST_DIR="$HOME/nn-lab"
NOTEBOOKS=(
  "notebooks/lab_intro_to_pytorch.ipynb"
)

echo "==> neural-net lab setup"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found. install Anaconda or Miniconda first, then re-run."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

echo "==> installing graphviz (system 'dot' + python binding) via conda-forge"
conda install -y -c conda-forge graphviz python-graphviz

echo "==> installing python libraries"
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn torch torchvision torchviz torchview jupyterlab

echo "==> downloading lab notebook(s) to $DEST_DIR"
mkdir -p "$DEST_DIR"
for nb in "${NOTEBOOKS[@]}"; do
  fname="$(basename "$nb")"
  curl -fsSL "$REPO_RAW_BASE/$nb" -o "$DEST_DIR/$fname"
  echo "    fetched $fname"
done

echo "==> verifying 'dot' is on PATH"
dot -V || echo "warning: dot not found on PATH yet - restart your terminal/kernel"

cat <<EOF

all set. to start the lab:
  conda activate $ENV_NAME
  cd $DEST_DIR
  jupyter lab
EOF
