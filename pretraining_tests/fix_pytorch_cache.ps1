# Stop Python processes
Write-Host "Stopping Python processes..."
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force

# Clean all cache
Write-Host "Cleaning PyTorch cache..."
$locations = @(
    "$env:TEMP\torchinductor_*",
    "$env:LOCALAPPDATA\Temp\torchinductor_*", 
    "$env:USERPROFILE\.cache\torch",
    "$env:USERPROFILE\.triton\cache"
)

foreach ($loc in $locations) {
    Remove-Item -Recurse -Force $loc -ErrorAction SilentlyContinue
}

# Create fresh cache
$customCache = "C:\hrm_pytorch_cache"
Write-Host "Creating custom cache at $customCache"
New-Item -ItemType Directory -Force -Path $customCache | Out-Null

# Set environment variables
$env:TORCHINDUCTOR_CACHE_DIR = $customCache
$env:TRITON_CACHE_DIR = "$customCache\triton"
$env:TORCH_COMPILE_DEBUG = "1"

Write-Host "Cache cleaned and configured!"
Write-Host "Custom cache: $customCache"