param(
    [string[]]$Datasets = @("benchmark", "beta"),
    [int[]]$Seeds = @(20260929, 20260930, 20260931),
    [int]$FoldCount = 5,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$config = Join-Path $PSScriptRoot "..\configs\paper_multidataset.yaml"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        for ($fold = 0; $fold -lt $FoldCount; $fold++) {
            $runDir = Join-Path $projectRoot (
                "runs\paper_v24\short_window_schedule_formal\{0}\harmonic_fold_v4_1\w04x2\seed-{1}\fold-{2}" -f
                $dataset, $seed, $fold
            )
            & $python -m src.paper_train `
                --config $config `
                --dataset $dataset `
                --architecture harmonic_fold_v4_1 `
                --mode full `
                --seed $seed `
                --fold-index $fold `
                --fold-count $FoldCount `
                --epochs 60 `
                --device $Device `
                --train-windows 0.4 0.4 0.6 0.8 1.0 1.2 `
                --run-dir $runDir | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed: dataset=$dataset seed=$seed fold=$fold"
            }
            Write-Output ("complete dataset={0} seed={1} fold={2}" -f $dataset, $seed, $fold)
        }
    }
}
