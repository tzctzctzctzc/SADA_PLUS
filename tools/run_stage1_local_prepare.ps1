param(
    [int]$WaitForPid = 0,
    [string]$DataRoot = "F:\SADA"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "D:\ANACONDA\python.exe"
$videoRoot = Join-Path $DataRoot "EPIC_KITCHENS\videos"
$manifestRoot = Join-Path $DataRoot "matrix_manifests"
$splitFile = Join-Path $DataRoot "epic_55_splits.csv"
$md5File = Join-Path $DataRoot "epic_md5.csv"
$frameRoot = Join-Path $DataRoot "matrix_frames_16"
$stageFile = Join-Path $DataRoot "download_priority_stage.txt"
$repairIds = @("P01_09", "P03_05", "P06_07", "P06_09", "P12_03", "P24_08")

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
Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_MD5_REPAIR"
$repair = Start-Process -FilePath $python -WindowStyle Hidden -Wait -PassThru `
    -ArgumentList (@(
        "-u", "-m", "tools.download_epic_manifest_videos",
        "--scenario-root", $manifestRoot,
        "--video-root", $videoRoot,
        "--epic55-splits", $splitFile,
        "--md5-csv", $md5File,
        "--verify-cached-md5",
        "--workers", "6", "--connections", "8", "--retries", "3",
        "--shifts", "S5", "--video-ids"
    ) + $repairIds) `
    -RedirectStandardOutput (Join-Path $DataRoot "repair_S5_source_md5_20260911.stdout.log") `
    -RedirectStandardError (Join-Path $DataRoot "repair_S5_source_md5_20260911.stderr.log")
if ($repair.ExitCode -ne 0) {
    Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_MD5_REPAIR_FAILED exit=$($repair.ExitCode)"
    exit $repair.ExitCode
}

Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_16_FRAME_EXTRACTION"
$prepare = Start-Process -FilePath $python -WindowStyle Hidden -Wait -PassThru `
    -ArgumentList @(
        "-u", "-m", "tools.prepare_vlm_frames",
        "--scenario-root", $manifestRoot,
        "--frame-cache", $frameRoot,
        "--video-root", $videoRoot,
        "--epic55-splits", $splitFile,
        "--frame-count", "16", "--shifts", "S5", "--parts", "source",
        "--workers", "2"
    ) `
    -RedirectStandardOutput (Join-Path $DataRoot "prepare_S5_source_16frames_retry_20260911.stdout.log") `
    -RedirectStandardError (Join-Path $DataRoot "prepare_S5_source_16frames_retry_20260911.stderr.log")
if ($prepare.ExitCode -ne 0) {
    Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_16_FRAME_EXTRACTION_FAILED exit=$($prepare.ExitCode)"
    exit $prepare.ExitCode
}

$sourceCache = Join-Path $frameRoot "source\s5_source"
$complete = @(
    Get-ChildItem -LiteralPath $sourceCache -Directory | Where-Object {
        $directory = $_.FullName
        -not (1..16 | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $directory ("frame_{0:D2}.jpg" -f $_)))
        })
    }
).Count
if ($complete -ne 200) {
    Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_16_FRAME_AUDIT_FAILED complete=$complete"
    exit 1
}

$archive = Join-Path $DataRoot "s5_source_16frames_20260911.tar"
& tar.exe -cf $archive -C $frameRoot "source"
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_16_ARCHIVE_FAILED exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

Set-Content -LiteralPath $stageFile -Value "S5_SOURCE_16_READY archive=$archive"

$followup = Start-Process -FilePath $python -WindowStyle Hidden -PassThru `
    -ArgumentList @(
        "-u", "-m", "tools.download_epic_manifest_videos",
        "--scenario-root", $manifestRoot,
        "--video-root", $videoRoot,
        "--epic55-splits", $splitFile,
        "--md5-csv", $md5File,
        "--workers", "16", "--connections", "8", "--retries", "3",
        "--shifts", "S1", "S2", "S6"
    ) `
    -RedirectStandardOutput (Join-Path $DataRoot "download_S1_S2_S6_followup_20260911.stdout.log") `
    -RedirectStandardError (Join-Path $DataRoot "download_S1_S2_S6_followup_20260911.stderr.log")
Set-Content -LiteralPath (Join-Path $DataRoot "download_S1S6_priority.pid") -Value $followup.Id
