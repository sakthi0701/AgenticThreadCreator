
# prepare_frontend.ps1

$distDir = "dist"

# Create dist directory if it doesn't exist
if (-not (Test-Path -Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir | Out-Null
    Write-Host "Created $distDir directory." -ForegroundColor Green
} else {
    Write-Host "Cleaning existing $distDir directory..." -ForegroundColor Yellow
    Remove-Item -Path "$distDir\*" -Force -Recurse
}

# Copy files
Copy-Item "index.html" -Destination $distDir
Copy-Item "favicon.ico" -Destination $distDir
# Copy favicon.png if it exists as fallback (index.html references .ico but usually good to have both if linked)
if (Test-Path "favicon.png") {
    Copy-Item "favicon.png" -Destination $distDir
}

Write-Host "✅ Frontend files ready in '$distDir' folder!" -ForegroundColor Cyan
Write-Host "👉 Next step: Drag and drop the '$distDir' folder into Netlify." -ForegroundColor White
