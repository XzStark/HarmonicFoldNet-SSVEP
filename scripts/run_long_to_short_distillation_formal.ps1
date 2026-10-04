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
            $teacherDir = Join-Path $projectRoot (
                "runs\paper_v26\long_teacher\{0}\harmonic_fold_v4_1\full\seed-{1}\fold-{2}" -f
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
                --train-windows 1.2 `
                --selection-windows 1.2 `
                --save-checkpoint `
                --run-dir $teacherDir | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Teacher training failed: dataset=$dataset seed=$seed fold=$fold"
            }

            $studentDir = Join-Path $projectRoot (
                "runs\paper_v26\long_to_short_formal\{0}\harmonic_fold_v4_1\full\seed-{1}\fold-{2}" -f
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
                --teacher-checkpoint (Join-Path $teacherDir "model.pt") `
                --teacher-window 1.2 `
                --distill-windows 0.4 0.6 `
                --distillation-weight 0.5 `
                --distillation-temperature 2.0 `
                --run-dir $studentDir | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Student training failed: dataset=$dataset seed=$seed fold=$fold"
            }
            Write-Output ("complete dataset={0} seed={1} fold={2}" -f $dataset, $seed, $fold)
        }
    }
}
