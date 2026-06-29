$RepoRawBase   = "https://raw.githubusercontent.com/cillustrisimo/ydl_2026/main"
$EnvName       = "nn-lab"
$PythonVersion = "3.11"
$DestDir       = "$HOME\nn-lab"
$Notebooks     = @(
  "notebooks/lab_intro_to_pytorch.ipynb"
)

$ErrorActionPreference = "Stop"
Write-Host "==> neural-net lab setup"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Error "conda not found. open an Anaconda PowerShell Prompt (or install Anaconda/Miniconda), then re-run."
  return
}

$envs = conda env list | Out-String
if ($envs -notmatch "(^|[\\/\s])$([regex]::Escape($EnvName))(\s|$)") {
  Write-Host "==> creating conda env '$EnvName' (python $PythonVersion)"
  conda create -y -n $EnvName "python=$PythonVersion"
}

Write-Host "==> installing graphviz (system 'dot' + python binding) via conda-forge"
conda install -y -n $EnvName -c conda-forge graphviz python-graphviz

Write-Host "==> installing python libraries"
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install numpy matplotlib scikit-learn torch torchvision torchviz torchview jupyterlab

Write-Host "==> downloading lab notebook(s) to $DestDir"
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
foreach ($nb in $Notebooks) {
  $fname = Split-Path $nb -Leaf
  Invoke-WebRequest -UseBasicParsing -Uri "$RepoRawBase/$nb" -OutFile "$DestDir\$fname"
  Write-Host "    fetched $fname"
}

Write-Host "==> verifying 'dot'"
try { conda run -n $EnvName dot -V } catch { Write-Host "warning: dot not found yet - restart your terminal/kernel" }

Write-Host ""
Write-Host "all set. to start the lab:"
Write-Host "  conda activate $EnvName"
Write-Host "  cd $DestDir"
Write-Host "  jupyter lab"
