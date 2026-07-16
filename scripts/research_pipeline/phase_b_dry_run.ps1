param(
  [string]$RepositoryRoot = (Get-Location).Path,
  [string]$StrategyName = "phase-b-dry-run-$([guid]::NewGuid().ToString('N').Substring(0, 8))",
  [switch]$AutomatedTest,
  [switch]$Detached
)

$root = (Resolve-Path $RepositoryRoot).Path
$env:PYTHONPATH = Join-Path $root "src"

if ($AutomatedTest) {
  # Fixture mode is deliberately not a Smithers run. It uses a unique temp
  # registry and the Python bridge only; no resumable run or shared registry.
  $registry = Join-Path ([IO.Path]::GetTempPath()) "phase-b-fixture-$([guid]::NewGuid().ToString('N')).sqlite3"
  $env:RESEARCH_PIPELINE_REGISTRY = $registry
  try {
    & py -m pytest -q (Join-Path $root "tests/research_pipeline/test_phase_b_core.py")
    exit $LASTEXITCODE
  } finally {
    Remove-Item -LiteralPath $registry -Force -ErrorAction SilentlyContinue
  }
}

$env:RESEARCH_PIPELINE_REGISTRY = Join-Path $root "research_registry\dry-run-$([guid]::NewGuid().ToString('N')).sqlite3"
$input = [ordered]@{
  strategy_name = $StrategyName
  natural_language_description = "A fictional deterministic entry and exit rule for a workflow fixture."
  requested_markets = @("TEST")
  requested_timeframes = @("1h")
  repository_root = $root
  registry_path = $env:RESEARCH_PIPELINE_REGISTRY
  dry_run = $true
  implementation_enabled = $false
} | ConvertTo-Json -Compress
$smithers = (Resolve-Path (Join-Path $root ".smithers/node_modules/.bin/smithers.exe")).Path
$arguments = @("up", ".smithers/workflows/trading-research-phase-b.tsx", "--no-report", "--input=$input")
if ($Detached) { $arguments += "-d" }
$psi = [Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $smithers
$psi.WorkingDirectory = $root
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.Arguments = ($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
$process = [Diagnostics.Process]::Start($psi)
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()
$output = @($stdout -split "`r?`n")
$output
if ($stderr) { [Console]::Error.WriteLine($stderr) }
if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3) { exit $process.ExitCode }

# This script intentionally stops here. Approval and resume are operator
# actions against the printed run; no bridge command is invoked while paused.
$runId = ($output | Select-String -Pattern "(?:run-[0-9]+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})" | Select-Object -First 1).Matches.Value
if ($runId) {
  Write-Output "RUN_ID=$runId"
  Write-Output "APPROVAL_COMMAND=smithers approve $runId --node approve-spec --by operator"
  Write-Output "RESUME_COMMAND=smithers up .smithers/workflows/trading-research-phase-b.tsx --run-id $runId --resume true"
}
