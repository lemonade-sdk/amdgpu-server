<#
Generates an MSVC import library (onnxruntime-genai.lib) from a DLL that ships
without one. The AMD GPU package distributes onnxruntime-genai.dll but no .lib,
so we synthesize one from its export table: dumpbin /exports -> .def -> lib.exe.
#>
param(
    [Parameter(Mandatory = $true)][string]$Dll,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][string]$Dumpbin,
    [Parameter(Mandatory = $true)][string]$Lib,
    [string]$Machine = "x64"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dll))     { throw "DLL not found: $Dll" }
if (-not (Test-Path $Dumpbin)) { throw "dumpbin not found: $Dumpbin" }
if (-not (Test-Path $Lib))     { throw "lib.exe not found: $Lib" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$dllName = [System.IO.Path]::GetFileName($Dll)
$defPath = Join-Path $OutDir "onnxruntime-genai.def"
$libPath = Join-Path $OutDir "onnxruntime-genai.lib"

Write-Host "Generating import lib for $dllName ..."

$exports = & $Dumpbin /exports $Dll
$names = @()
foreach ($line in $exports) {
    # Match: <ordinal> <hint(hex)> <RVA(hex)> <name>
    if ($line -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
        $names += $Matches[1]
    }
}

if ($names.Count -eq 0) { throw "No exports parsed from $dllName" }

$def = New-Object System.Text.StringBuilder
[void]$def.AppendLine("LIBRARY $dllName")
[void]$def.AppendLine("EXPORTS")
foreach ($n in $names) { [void]$def.AppendLine("    $n") }
Set-Content -Path $defPath -Value $def.ToString() -Encoding ascii

& $Lib "/def:$defPath" "/machine:$Machine" "/out:$libPath" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "lib.exe failed with exit code $LASTEXITCODE" }

Write-Host "Wrote $libPath ($($names.Count) exports)"
