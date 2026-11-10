param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)

# Native Microsoft Word layout/field update, launched as a separate invisible instance.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($InputPath, $false, $false)
    $document.Repaginate()
    $null = $document.Fields.Update()
    $document.Save()
    $document.ExportAsFixedFormat($OutputPath, 17)
    Write-Output ('Pages: ' + $document.ComputeStatistics(2))
    Write-Output ('PDF: ' + $OutputPath)
}
finally {
    if ($null -ne $document) {
        [object]$closeOption = 0
        $document.Close([ref]$closeOption)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        [object]$quitOption = 0
        $word.Quit([ref]$quitOption)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
