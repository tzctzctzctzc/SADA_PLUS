param(
    [int]$WaitForPid = 0,
    [string]$DataRoot = "F:\SADA"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "D:\ANACONDA\python.exe"
$videoRoot = Join-Path $DataRoot "EPIC_KITCHENS\videos"
$manifestRoot = Join-Path $DataRoot "matrix_manifests"
$frameRoot = Join-Path $DataRoot "frames"
$splitFile = Join-Path $DataRoot "epic_55_splits.csv"
$md5File = Join-Path $DataRoot "epic_md5.csv"
$statusFile = Join-Path $DataRoot "s3_s4_vlm_pipeline.status.txt"

$env:HTTP_PROXY = $null
$env:HTTPS_PROXY = $null
$env:ALL_PROXY = $null
$env:NO_PROXY = "*"

if ($WaitForPid -gt 0) {
    $process = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($process) {
        $process.WaitForExit()
    }
}

Set-Location -LiteralPath $repo
Set-Content -LiteralPath $statusFile -Value "VERIFY_AND_REPAIR"
$verify = Start-Process -FilePath $python -WindowStyle Hidden -Wait -PassThru `
    -ArgumentList @(
        "-u", "-m", "tools.download_epic_manifest_videos",
        "--scenario-root", $manifestRoot,
        "--video-root", $videoRoot,
        "--epic55-splits", $splitFile,
        "--md5-csv", $md5File,
        "--verify-cached-md5",
        "--workers", "3", "--connections", "8", "--retries", "3",
        "--shifts", "S3", "S4"
    ) `
    -RedirectStandardOutput (Join-Path $DataRoot "verify_repair_S3_S4_20260912.stdout.log") `
    -RedirectStandardError (Join-Path $DataRoot "verify_repair_S3_S4_20260912.stderr.log")
if ($verify.ExitCode -ne 0) {
    Set-Content -LiteralPath $statusFile -Value "VERIFY_AND_REPAIR_FAILED exit=$($verify.ExitCode)"
    exit $verify.ExitCode
}

Set-Content -LiteralPath $statusFile -Value "EXTRACT_8_FRAMES"
$prepare = Start-Process -FilePath $python -WindowStyle Hidden -Wait -PassThru `
    -ArgumentList @(
        "-u", "-m", "tools.prepare_vlm_frames",
        "--scenario-root", $manifestRoot,
        "--frame-cache", $frameRoot,
        "--video-root", $videoRoot,
        "--epic55-splits", $splitFile,
        "--frame-count", "8", "--shifts", "S3", "S4",
        "--parts", "source", "target", "--workers", "2"
    ) `
    -RedirectStandardOutput (Join-Path $DataRoot "prepare_S3_S4_8frames_verified_20260912.stdout.log") `
    -RedirectStandardError (Join-Path $DataRoot "prepare_S3_S4_8frames_verified_20260912.stderr.log")
if ($prepare.ExitCode -ne 0) {
    Set-Content -LiteralPath $statusFile -Value "EXTRACT_8_FRAMES_FAILED exit=$($prepare.ExitCode)"
    exit $prepare.ExitCode
}

Set-Content -LiteralPath $statusFile -Value "ARCHIVE_FRAMES"
$archive = Join-Path $DataRoot "s3_s4_matrix_frames_8_20260912.tar"
& tar.exe -cf $archive -C $frameRoot `
    "source/s3_source" "source/s4_source" "target/s3" "target/s4"
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath $statusFile -Value "ARCHIVE_FRAMES_FAILED exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

Set-Content -LiteralPath $statusFile -Value "READY archive=$archive"
