# PowerShell script to setup conda environment for FER project
# Run: .\scripts\setup_env.ps1

Write-Host "=================================================="
Write-Host "Setting up Lightweight Multimodal FER Environment"
Write-Host "=================================================="

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
}

# Environment name
$ENV_NAME = "fer"

# Check if environment already exists
$envExists = conda env list | Select-String -Pattern "^$ENV_NAME\s"
if ($envExists) {
    Write-Host "Environment '$ENV_NAME' already exists."
    $response = Read-Host "Do you want to remove and recreate it? (y/n)"
    if ($response -eq 'y') {
        Write-Host "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    } else {
        Write-Host "Updating existing environment..."
        conda env update -n $ENV_NAME -f environment.yml
        Write-Host ""
        Write-Host "Done! Activate with: conda activate $ENV_NAME"
        exit 0
    }
}

Write-Host ""
Write-Host "Creating conda environment '$ENV_NAME'..."
Write-Host "This may take 10-15 minutes..."
Write-Host ""

# Create environment from yml
conda env create -f environment.yml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "SUCCESS! Environment created."
    Write-Host "=================================================="
    Write-Host ""
    Write-Host "To activate: conda activate $ENV_NAME"
    Write-Host ""
    Write-Host "To test installation:"
    Write-Host "  python -c `"import torch; print(f'PyTorch: {torch.__version__}')`""
    Write-Host "  python -c `"import transformers; print(f'Transformers: {transformers.__version__}')`""
    Write-Host "  python tests/test_visual_branch.py"
    Write-Host ""
    Write-Host "To download SigLIP2 model:"
    Write-Host "  python -c `"from transformers import AutoModel; AutoModel.from_pretrained('google/siglip2-base-patch16-224')`""
} else {
    Write-Host ""
    Write-Host "ERROR: Failed to create environment."
    Write-Host "Check the error messages above."
    exit 1
}

