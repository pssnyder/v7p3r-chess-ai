# Simple Git LFS Analysis for v7p3r-chess-ai
# Provides actionable recommendations

$repoPath = "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai"
cd $repoPath

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " Git LFS Analysis for v7p3r-chess-ai" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if this is a git repo
if (Test-Path ".git") {
    Write-Host "[OK] Git repository detected" -ForegroundColor Green
} else {
    Write-Host "[WARN] Not a Git repository yet - run 'git init' first" -ForegroundColor Yellow
}

# 2. Check Git LFS
try {
    $lfsVersion = git lfs version 2>&1 | Out-String
    Write-Host "[OK] Git LFS installed: $($lfsVersion.Trim())" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Git LFS not installed" -ForegroundColor Yellow
    Write-Host "   Install from: https://git-lfs.github.com/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "------- CURRENT REPOSITORY STATE -------" -ForegroundColor Cyan
Write-Host ""

# 3. Total repository size
$allFiles = Get-ChildItem -Recurse -File | Where-Object {$_.FullName -notmatch '\\\.git\\'}
$totalSize = ($allFiles | Measure-Object -Property Length -Sum).Sum / 1GB

Write-Host "Total files: $($allFiles.Count)"
Write-Host "Total size:  $([math]::Round($totalSize, 2)) GB" -ForegroundColor Yellow
Write-Host ""

# 4. Breakdown by size category
$large = $allFiles | Where-Object {$_.Length -gt 100MB}
$medium = $allFiles | Where-Object {$_.Length -gt 1MB -and $_.Length -le 100MB}
$small = $allFiles | Where-Object {$_.Length -le 1MB}

Write-Host "Size breakdown:"
Write-Host "  Large files (>100 MB):  $($large.Count) files, $([math]::Round(($large | Measure-Object Length -Sum).Sum / 1GB, 2)) GB" -ForegroundColor Red
Write-Host "  Medium files (1-100 MB): $($medium.Count) files, $([math]::Round(($medium | Measure-Object Length -Sum).Sum / 1GB, 2)) GB" -ForegroundColor Yellow
Write-Host "  Small files (<1 MB):     $($small.Count) files, $([math]::Round(($small | Measure-Object Length -Sum).Sum / 1GB, 2)) GB" -ForegroundColor Green
Write-Host ""

# 5. Top 10 largest files
Write-Host "------- TOP 10 LARGEST FILES -------" -ForegroundColor Cyan
$allFiles | Sort-Object Length -Descending | Select-Object -First 10 | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    $path = $_.FullName.Replace($repoPath + '\', '')
    Write-Host "  $sizeMB MB - $path"
}
Write-Host ""

# 6. Directory breakdown
Write-Host "------- SIZE BY DIRECTORY -------" -ForegroundColor Cyan
Get-ChildItem -Directory | ForEach-Object {
    $dirSize = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Directory = $_.Name
        SizeGB = [math]::Round($dirSize / 1GB, 2)
    }
} | Sort-Object SizeGB -Descending | Format-Table -AutoSize
Write-Host ""

# 7. File type analysis
Write-Host "------- LARGEST FILE TYPES -------" -ForegroundColor Cyan
$allFiles | Group-Object Extension | ForEach-Object {
    $typeSize = ($_.Group | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Type = if($_.Name){"*" + $_.Name}else{"NoExt"}
        Count = $_.Count
        SizeGB = [math]::Round($typeSize / 1GB, 3)
    }
} | Where-Object {$_.SizeGB -gt 0.1} | Sort-Object SizeGB -Descending | Format-Table -AutoSize
Write-Host ""

# 8. Recommendations
Write-Host "------- RECOMMENDATIONS -------" -ForegroundColor Cyan
Write-Host ""

$oldVersions = @('v3.0', 'v4.0', 'v5.0')
$oldVersionSize = 0
foreach ($dir in $oldVersions) {
    if (Test-Path $dir) {
        $dirSize = (Get-ChildItem $dir -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $oldVersionSize += $dirSize
    }
}

if ($oldVersionSize -gt 0) {
    Write-Host "1. ARCHIVE OLD VERSIONS (RECOMMENDED)" -ForegroundColor Yellow
    Write-Host "   Directories: v3.0, v4.0, v5.0"
    Write-Host "   Total size: $([math]::Round($oldVersionSize / 1GB, 1)) GB ($([math]::Round($oldVersionSize / $totalSize / 1GB * 100, 0))% of repo)"
    Write-Host "   Action: Move to external archive folder"
    Write-Host "   Savings: ~$([math]::Round($oldVersionSize / 1GB, 1)) GB"
    Write-Host ""
}

Write-Host "2. GIT LFS STRATEGY" -ForegroundColor Yellow
Write-Host ""
Write-Host "   Track with Git LFS (large, irreplaceable):"
Write-Host "     - *.jsonl (raw training data)"
Write-Host "     - *.csv (sentiment analysis results)"
Write-Host "     - *.db (core databases)"
Write-Host ""
Write-Host "   Ignore (reproduceable outputs):"
Write-Host "     - *.pt, *.pth (model checkpoints)"
Write-Host "     - *.pkl, *.npy (preprocessed arrays)"
Write-Host "     - *.exe (executables)"
Write-Host ""

# 9. Cost estimate
$jsonlSize = ($allFiles | Where-Object {$_.Extension -eq '.jsonl'} | Measure-Object -Property Length -Sum).Sum / 1GB
$csvSize = ($allFiles | Where-Object {$_.Extension -eq '.csv'} | Measure-Object -Property Length -Sum).Sum / 1GB
$dbSize = ($allFiles | Where-Object {$_.Extension -eq '.db'} | Measure-Object -Property Length -Sum).Sum / 1GB

$lfsSize = $jsonlSize + $csvSize + $dbSize
$lfsAfterArchive = $lfsSize - ($oldVersionSize / 1GB)

Write-Host "3. COST ESTIMATES" -ForegroundColor Yellow
Write-Host ""
Write-Host "   Scenario A: Track everything with LFS"
Write-Host "     Size: $([math]::Round($lfsSize, 1)) GB"
Write-Host "     Cost: ~$([math]::Ceiling($lfsSize / 50) * 5) USD/month"
Write-Host ""
Write-Host "   Scenario B: After archiving old versions"
Write-Host "     Size: $([math]::Round($lfsAfterArchive, 1)) GB"
Write-Host "     Cost: ~$([math]::Ceiling($lfsAfterArchive / 50) * 5) USD/month" -ForegroundColor Green
Write-Host ""
Write-Host "   Scenario C: v7.0 only + critical v6.0 data"
Write-Host "     Size: ~5-10 GB (estimated)"
Write-Host "     Cost: FREE to 5 USD/month" -ForegroundColor Green
Write-Host ""

Write-Host "------- NEXT STEPS -------" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review GIT_LFS_STRATEGY.md for detailed plan"
Write-Host "2. Archive v3.0, v4.0, v5.0 to external storage"
Write-Host "3. Update .gitignore to exclude reproduceable files"
Write-Host "4. Create .gitattributes for Git LFS patterns"
Write-Host "5. Run 'git lfs track' commands"
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Export summary
$summary = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    totalFiles = $allFiles.Count
    totalSizeGB = [math]::Round($totalSize, 2)
    oldVersionsSizeGB = [math]::Round($oldVersionSize / 1GB, 2)
    recommendedLFSSizeGB = [math]::Round($lfsAfterArchive, 2)
    estimatedCostUSD = [math]::Ceiling($lfsAfterArchive / 50) * 5
}

$summary | ConvertTo-Json | Out-File "git_lfs_analysis.json"
Write-Host "[SAVED] Analysis exported to: git_lfs_analysis.json" -ForegroundColor Green
Write-Host ""
