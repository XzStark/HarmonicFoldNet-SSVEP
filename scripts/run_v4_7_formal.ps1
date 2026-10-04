param(
    [string[]]$Datasets = @("beta"),
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
                "runs\paper_v33\v4_7_formal\{0}\harmonic_fold_v4_7\full\seed-{1}\fold-{2}" -f
                $dataset, $seed, $fold
            )
            & $python -m src.paper_train `
                --config $config `
                --dataset $dataset `
                --architecture harmonic_fold_v4_7 `
                --mode full `
                --seed $seed `
                --fold-index $fold `
                --fold-count $FoldCount `
                --epochs 60 `
                --device $Device `
                --run-dir $runDir | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed: dataset=$dataset seed=$seed fold=$fold"
            }
            Write-Output ("complete dataset={0} seed={1} fold={2}" -f $dataset, $seed, $fold)
        }
    }
}
