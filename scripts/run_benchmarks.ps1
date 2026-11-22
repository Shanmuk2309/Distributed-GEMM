param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResultsDir = Join-Path $ProjectRoot 'results'
$LogDir = Join-Path $ResultsDir 'benchmark_logs'
$Compiler = 'mpicc'
$Mpiexec = 'mpiexec'
$IncludePath = 'C:/msys64/ucrt64/include/openblas'
$LibraryPath = 'C:/msys64/ucrt64/lib'
$Sizes = @(512, 1024, 2048, 4096)
$ProcessCounts = @(1, 4, 9, 16)
$CsvColumns = @(
    'Matrix_Size', 'Processes', 'Precision', 'Execution_Time',
    'Communication_Time', 'Computation_Time', 'GFLOPS', 'C00',
    'Max_Absolute_Error', 'Max_Relative_Error', 'Verification'
)
$Programs = @(
    [pscustomobject]@{ Name = 'oned_gemm'; Source = 'src/oned_gemm.c'; Executable = 'oned_gemm.exe'; Csv = 'oned_gemm_float.csv'; Precision = 'Single' },
    [pscustomobject]@{ Name = 'oned_gemm_double'; Source = 'src/oned_gemm_double.c'; Executable = 'oned_gemm_double.exe'; Csv = 'oned_gemm_double.csv'; Precision = 'Double' },
    [pscustomobject]@{ Name = 'summa'; Source = 'src/summa.c'; Executable = 'summa.exe'; Csv = 'summa_float.csv'; Precision = 'Single' },
    [pscustomobject]@{ Name = 'summa_double'; Source = 'src/summa_double.c'; Executable = 'summa_double.exe'; Csv = 'summa_double.csv'; Precision = 'Double' },
    [pscustomobject]@{ Name = 'cannon'; Source = 'src/cannon.c'; Executable = 'cannon.exe'; Csv = 'cannon_float.csv'; Precision = 'Single' },
    [pscustomobject]@{ Name = 'cannon_double'; Source = 'src/cannon_double.c'; Executable = 'cannon_double.exe'; Csv = 'cannon_double.csv'; Precision = 'Double' },
    [pscustomobject]@{ Name = 'summa_overlap'; Source = 'src/summa_overlap.c'; Executable = 'summa_overlap.exe'; Csv = 'summa_overlap_float.csv'; Precision = 'Single' },
    [pscustomobject]@{ Name = 'summa_overlap_double'; Source = 'src/summa_overlap_double.c'; Executable = 'summa_overlap_double.exe'; Csv = 'summa_overlap_double.csv'; Precision = 'Double' },
    [pscustomobject]@{ Name = 'cannon_overlap'; Source = 'src/cannon_overlap.c'; Executable = 'cannon_overlap.exe'; Csv = 'cannon_overlap_float.csv'; Precision = 'Single' },
    [pscustomobject]@{ Name = 'cannon_overlap_double'; Source = 'src/cannon_overlap_double.c'; Executable = 'cannon_overlap_double.exe'; Csv = 'cannon_overlap_double.csv'; Precision = 'Double' }
)

function Write-CapturedNativeRun {
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$StdOutPath,
        [Parameter(Mandatory)][string]$StdErrPath
    )

    # Native output is redirected to separate complete logs. The process is
    # synchronous: the next MPI configuration cannot start until it exits.
    & $Executable @Arguments 1> $StdOutPath 2> $StdErrPath
    return $LASTEXITCODE
}

function Get-NumberFromOutput {
    param([string]$Text, [string]$Pattern, [string]$Label)
    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) { throw "Missing or invalid metric: $Label" }
    $parsed = 0.0
    if (-not [double]::TryParse($match.Groups['value'].Value, [System.Globalization.NumberStyles]::Float, $Invariant, [ref]$parsed)) {
        throw "Could not parse metric: $Label"
    }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) { throw "Non-finite metric: $Label" }
    return $parsed
}

function Format-Number {
    param([double]$Value)
    return $Value.ToString('R', $Invariant)
}

function Get-SuccessfulExistingRecord {
    param([string]$CsvPath, [int]$N, [int]$P, [string]$Precision)
    if (-not (Test-Path -LiteralPath $CsvPath)) { throw "Required results CSV is missing: $CsvPath" }
    $rows = Import-Csv -LiteralPath $CsvPath
    foreach ($row in $rows) {
        if ([string]$row.Matrix_Size -ne [string]$N -or
            [string]$row.Processes -ne [string]$P -or
            [string]$row.Precision -ne $Precision -or
            [string]$row.Verification -notmatch '^PASS(?:ED)?$') { continue }
        $rel = 0.0
        if ([double]::TryParse([string]$row.Max_Relative_Error, [System.Globalization.NumberStyles]::Float, $Invariant, [ref]$rel) -and $rel -lt 1e-4) {
            return $true
        }
    }
    return $false
}

function Append-CsvRecord {
    param([string]$CsvPath, [pscustomobject]$Record)
    $Record | Export-Csv -LiteralPath $CsvPath -Append -NoTypeInformation -Encoding utf8
}

function Write-FailureRecord {
    param([string]$Program, [int]$N, [int]$P, [string]$Precision, [string]$Message, [string]$StdOutPath, [string]$StdErrPath)
    $failurePath = Join-Path $ResultsDir 'benchmark_failures.csv'
    if (-not (Test-Path -LiteralPath $failurePath)) {
        'Program,Matrix_Size,Processes,Precision,Error,Timestamp,StdOut_Log,StdErr_Log' | Set-Content -LiteralPath $failurePath -Encoding utf8
    }
    $record = [pscustomobject]@{
        Program = $Program
        Matrix_Size = $N
        Processes = $P
        Precision = $Precision
        Error = $Message
        Timestamp = [DateTime]::UtcNow.ToString('o', $Invariant)
        StdOut_Log = [IO.Path]::GetRelativePath($ProjectRoot, $StdOutPath)
        StdErr_Log = [IO.Path]::GetRelativePath($ProjectRoot, $StdErrPath)
    }
    $record | Export-Csv -LiteralPath $failurePath -Append -NoTypeInformation -Encoding utf8
}

function Get-Median {
    param([double[]]$Values)
    $ordered = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($ordered.Count / 2)
    if (($ordered.Count % 2) -eq 1) { return [double]$ordered[$middle] }
    return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0
}

function Write-SummaryFiles {
    param([int]$Attempted, [int]$Succeeded, [int]$Failed, [int]$Skipped, [object[]]$RunFailures)

    $summaryRows = [System.Collections.Generic.List[object]]::new()
    foreach ($program in $Programs) {
        $csvPath = Join-Path $ResultsDir $program.Csv
        foreach ($group in (Import-Csv -LiteralPath $csvPath | Where-Object { $_.Precision -eq $program.Precision -and $_.Verification -match '^PASS(?:ED)?$' } | Group-Object Matrix_Size,Processes,Precision)) {
            $valid = @($group.Group | Where-Object {
                $rel = 0.0
                [double]::TryParse([string]$_.Max_Relative_Error, [System.Globalization.NumberStyles]::Float, $Invariant, [ref]$rel) -and $rel -lt 1e-4
            })
            if ($valid.Count -eq 0) { continue }
            $summaryRows.Add([pscustomobject]@{
                Program = $program.Name
                Matrix_Size = [int]$valid[0].Matrix_Size
                Processes = [int]$valid[0].Processes
                Precision = $program.Precision
                Successful_Runs = $valid.Count
                Median_Execution_Time = Format-Number (Get-Median @($valid | ForEach-Object { [double]::Parse($_.Execution_Time, $Invariant) }))
                Median_Communication_Time = Format-Number (Get-Median @($valid | ForEach-Object { [double]::Parse($_.Communication_Time, $Invariant) }))
                Median_Computation_Time = Format-Number (Get-Median @($valid | ForEach-Object { [double]::Parse($_.Computation_Time, $Invariant) }))
                Median_GFLOPS = Format-Number (Get-Median @($valid | ForEach-Object { [double]::Parse($_.GFLOPS, $Invariant) }))
            })
        }
    }

    $summaryCsv = Join-Path $ResultsDir 'benchmark_summary.csv'
    if ($summaryRows.Count -gt 0) { $summaryRows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding utf8 }
    else { 'Program,Matrix_Size,Processes,Precision,Successful_Runs,Median_Execution_Time,Median_Communication_Time,Median_Computation_Time,Median_GFLOPS' | Set-Content -LiteralPath $summaryCsv -Encoding utf8 }

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# MPI GEMM benchmark summary')
    $lines.Add('')
    $lines.Add("- Configurations attempted in this run: $Attempted")
    $lines.Add("- Successful configurations in this run: $Succeeded")
    $lines.Add("- Failed configurations in this run: $Failed")
    $lines.Add("- Skipped configurations with an existing successful row: $Skipped")
    $lines.Add("- Measurements: one run per configuration; summary medians therefore equal each configuration's single successful measurement unless pre-existing successful records are present.")
    $lines.Add('- Algorithms and precision variants: 1D GEMM, SUMMA, Cannon, SUMMA overlap, Cannon overlap; single and double precision.')
    $lines.Add('- Matrix sizes: 512, 1024, 2048, 4096. Matrix size 8192 was not run.')
    $lines.Add('- Process counts: 1, 4, 9, 16.')
    $lines.Add('- MPI startup preflight: 9 and 16 processes passed with hello_mpi.exe; OpenBLAS-linked 512 GEMM passed at 9 and 16 processes.')
    $lines.Add('- CSV files updated: successful rows appended to the mapped algorithm/precision CSVs; failures appended to benchmark_failures.csv when present.')
    $lines.Add('- All MPI benchmark jobs were run sequentially.')
    $lines.Add('')
    $lines.Add('## Successful configurations')
    $lines.Add('')
    $lines.Add('| Program | Precision | Successful matrix/process configurations |')
    $lines.Add('|---|---|---|')
    $script:relCheck = 0.0
    foreach ($program in $Programs) {
        $csvPath = Join-Path $ResultsDir $program.Csv
        $successRows = @(Import-Csv -LiteralPath $csvPath | Where-Object {
            $_.Precision -eq $program.Precision -and $_.Verification -match '^PASS(?:ED)?$' -and
            ([double]::TryParse([string]$_.Max_Relative_Error, [System.Globalization.NumberStyles]::Float, $Invariant, [ref]$script:relCheck) -and $script:relCheck -lt 1e-4)
        })
        $cells = @($successRows | ForEach-Object { "{0}/P{1}" -f $_.Matrix_Size, $_.Processes } | Sort-Object -Unique)
        $lines.Add("| $($program.Name) | $($program.Precision) | $($cells -join ', ') |")
    }
    $lines.Add('')
    $lines.Add('## Failures')
    $lines.Add('')
    if ($RunFailures.Count -eq 0) { $lines.Add('No failed benchmark configurations in this run.') }
    else {
        $lines.Add('| Program | Size | Processes | Reason |')
        $lines.Add('|---|---:|---:|---|')
        foreach ($failure in $RunFailures) {
            $reason = ([string]$failure.Error -replace '\|', '\|' -replace '\r?\n', ' ').Trim()
            $lines.Add("| $($failure.Program) | $($failure.Matrix_Size) | $($failure.Processes) | $reason |")
        }
    }
    $lines.Add('')
    $lines.Add('- 9-process execution worked in preflight: yes.')
    $lines.Add('- 16-process execution worked in preflight: yes.')
    $lines.Add('- Incomplete configurations: see failed and skipped counts above; configurations not attempted due to an early preflight/compiler stop are not included.')
    $lines | Set-Content -LiteralPath (Join-Path $ResultsDir 'benchmark_summary.md') -Encoding utf8
}

# Configure the required runtime before compiling, preflight, or launching MPI.
$env:Path = "C:\msys64\ucrt64\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:GOTO_NUM_THREADS = '1'
$env:OMP_DYNAMIC = 'FALSE'

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Validate and preserve the exact existing per-program CSV headers.
foreach ($program in $Programs) {
    $path = Join-Path $ResultsDir $program.Csv
    if (-not (Test-Path -LiteralPath $path)) { throw "Required existing CSV is missing: $path" }
    $header = (Get-Content -LiteralPath $path -TotalCount 1) -join ''
    if ($header.TrimStart([char]0xFEFF) -ne ($CsvColumns -join ',')) { throw "Unexpected CSV header; refusing to alter $path" }
}

# Compile every implementation before any benchmark configuration starts.
foreach ($program in $Programs) {
    $sourcePath = Join-Path $ProjectRoot $program.Source
    if (-not (Test-Path -LiteralPath $sourcePath)) { throw "Source file is missing: $sourcePath" }
    $exePath = Join-Path $ProjectRoot $program.Executable
    Write-Host "Compiling $($program.Source)"
    & $Compiler -O2 $sourcePath "-I$IncludePath" "-L$LibraryPath" -lopenblas -lm -o $exePath
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed for $($program.Source) with exit code $LASTEXITCODE" }
}
$helloSource = Join-Path $ProjectRoot 'hello_mpi.c'
& $Compiler -O2 $helloSource -o (Join-Path $ProjectRoot 'hello_mpi.exe')
if ($LASTEXITCODE -ne 0) { throw "Compilation failed for hello_mpi.c with exit code $LASTEXITCODE" }

# Run sequential launcher and BLAS preflights. A failure aborts before the 160-case campaign.
$preflightStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ', $Invariant)
foreach ($p in @(9, 16)) {
    $stdout = Join-Path $LogDir "preflight_hello_p${p}_${preflightStamp}.stdout.log"
    $stderr = Join-Path $LogDir "preflight_hello_p${p}_${preflightStamp}.stderr.log"
    $exit = Write-CapturedNativeRun -Executable $Mpiexec -Arguments @('-n', [string]$p, (Join-Path $ProjectRoot 'hello_mpi.exe')) -StdOutPath $stdout -StdErrPath $stderr
    $hello = Get-Content -LiteralPath $stdout -Raw
    if ($exit -ne 0 -or ([regex]::Matches($hello, 'Hello from process').Count -ne $p)) {
        throw "MPI startup preflight failed at $p processes (exit $exit). Logs: $stdout ; $stderr"
    }
    $stdout = Join-Path $LogDir "preflight_gemm_p${p}_${preflightStamp}.stdout.log"
    $stderr = Join-Path $LogDir "preflight_gemm_p${p}_${preflightStamp}.stderr.log"
    $exit = Write-CapturedNativeRun -Executable $Mpiexec -Arguments @('-n', [string]$p, (Join-Path $ProjectRoot 'oned_gemm.exe'), '512') -StdOutPath $stdout -StdErrPath $stderr
    $output = (Get-Content -LiteralPath $stdout -Raw) + "`n" + (Get-Content -LiteralPath $stderr -Raw)
    if ($exit -ne 0 -or $output -notmatch 'Verification\s*:\s*PASS(?:ED)?' -or $output -match 'Verification\s*:\s*FAIL(?:ED)?') {
        throw "OpenBLAS GEMM preflight failed at $p processes (exit $exit). Logs: $stdout ; $stderr"
    }
}

$attempted = 0
$succeeded = 0
$failed = 0
$skipped = 0
$runFailures = [System.Collections.Generic.List[object]]::new()
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ', $Invariant)

foreach ($n in $Sizes) {
    foreach ($p in $ProcessCounts) {
        foreach ($program in $Programs) {
            $csvPath = Join-Path $ResultsDir $program.Csv
            if (-not $Force -and (Get-SuccessfulExistingRecord -CsvPath $csvPath -N $n -P $p -Precision $program.Precision)) {
                $skipped++
                Write-Host "SKIP existing PASS: $($program.Name) N=$n P=$p"
                continue
            }

            $attempted++
            $stdoutPath = Join-Path $LogDir "${runStamp}_$($program.Name)_N${n}_P${p}.stdout.log"
            $stderrPath = Join-Path $LogDir "${runStamp}_$($program.Name)_N${n}_P${p}.stderr.log"
            $exePath = Join-Path $ProjectRoot $program.Executable
            Write-Host "RUN $($program.Name) N=$n P=$p"
            $exit = Write-CapturedNativeRun -Executable $Mpiexec -Arguments @('-n', [string]$p, $exePath, [string]$n) -StdOutPath $stdoutPath -StdErrPath $stderrPath
            $stdoutText = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { '' }
            $stderrText = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { '' }
            $combined = $stdoutText + "`n" + $stderrText

            try {
                if ($exit -ne 0) { throw "MPI executable exited with code $exit. $($stderrText.Trim())" }
                $matrix = [regex]::Match($stdoutText, 'Matrix size\s*:\s*(?<n>\d+)\s*x\s*\d+', 'IgnoreCase')
                if (-not $matrix.Success -or [int]$matrix.Groups['n'].Value -ne $n) { throw 'Program output did not report the requested matrix size.' }
                $procMatch = [regex]::Match($stdoutText, '(?:Number of processes|Processes)\s*:\s*(?<p>\d+)', 'IgnoreCase')
                if (-not $procMatch.Success -or [int]$procMatch.Groups['p'].Value -ne $p) { throw 'Program output did not report the requested process count.' }
                $verifyMatch = [regex]::Match($stdoutText, 'Verification\s*:\s*(?<status>PASS(?:ED)?|FAIL(?:ED)?)', 'IgnoreCase')
                if (-not $verifyMatch.Success) { throw 'OpenBLAS verification status is missing.' }
                if ($verifyMatch.Groups['status'].Value -notmatch '^PASS(?:ED)?$') { throw 'OpenBLAS verification reported failure.' }
                $execution = Get-NumberFromOutput $stdoutText 'Execution time\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'Execution_Time'
                $communication = Get-NumberFromOutput $stdoutText 'Communication time\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'Communication_Time'
                $computation = Get-NumberFromOutput $stdoutText 'Computation time\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'Computation_Time'
                $gflops = Get-NumberFromOutput $stdoutText '(?:Performance|GFLOP/s)\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'GFLOPS'
                $c00 = Get-NumberFromOutput $stdoutText 'C\[0\]\[0\]\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'C00'
                $maxAbs = Get-NumberFromOutput $stdoutText 'Maximum absolute error\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'Max_Absolute_Error'
                $maxRel = Get-NumberFromOutput $stdoutText 'Maximum relative error\s*:\s*(?<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)' 'Max_Relative_Error'
                if ($maxRel -ge 1e-4) { throw "Maximum relative error $maxRel is not below 1e-4." }
                if ($execution -le 0 -or $communication -lt 0 -or $computation -lt 0 -or $gflops -le 0) { throw 'One or more timing/performance metrics are outside valid ranges.' }

                $record = [pscustomobject]@{
                    Matrix_Size = $n
                    Processes = $p
                    Precision = $program.Precision
                    Execution_Time = Format-Number $execution
                    Communication_Time = Format-Number $communication
                    Computation_Time = Format-Number $computation
                    GFLOPS = Format-Number $gflops
                    C00 = Format-Number $c00
                    Max_Absolute_Error = Format-Number $maxAbs
                    Max_Relative_Error = Format-Number $maxRel
                    Verification = 'PASS'
                }
                Append-CsvRecord -CsvPath $csvPath -Record $record
                $succeeded++
                Write-Host "PASS $($program.Name) N=$n P=$p rel=$maxRel"
            }
            catch {
                $failed++
                $message = $_.Exception.Message
                $failure = [pscustomobject]@{ Program = $program.Name; Matrix_Size = $n; Processes = $p; Precision = $program.Precision; Error = $message }
                $runFailures.Add($failure)
                Write-FailureRecord -Program $program.Name -N $n -P $p -Precision $program.Precision -Message $message -StdOutPath $stdoutPath -StdErrPath $stderrPath
                Write-Warning "FAIL $($program.Name) N=$n P=$p : $message"
            }
        }
    }
}

Write-SummaryFiles -Attempted $attempted -Succeeded $succeeded -Failed $failed -Skipped $skipped -RunFailures $runFailures.ToArray()
Write-Host "Campaign complete. Attempted=$attempted Passed=$succeeded Failed=$failed Skipped=$skipped"
