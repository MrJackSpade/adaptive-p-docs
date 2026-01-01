# ============================================================================
# Adaptive-P Documentation Compiler (PowerShell)
# ============================================================================
# This script compiles all documentation sections into a single Documentation.md
# file with:
#   - Table of contents from README.md structure
#   - All section content concatenated in order  
#   - Images base64 encoded and embedded inline
#   - Sample files embedded inline where referenced (with anchor links)
#   - No external references - fully self-contained
# ============================================================================

param(
    [string]$OutputFile = "Documentation.md"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SectionsDir = Join-Path $ScriptDir "sections"
$ChartsDir = Join-Path $ScriptDir "charts"
$SamplesDir = Join-Path $ScriptDir "samples"
$OutputPath = Join-Path $ScriptDir $OutputFile

# Section files in order
$SectionFiles = @(
    "01_abstract.md",
    "02_introduction.md",
    "03_related_work.md",
    "04_algorithm.md",
    "05_parameters.md",
    "06_integration.md",
    "07_empirical_validation.md",
    "08_implementation.md",
    "09_conclusion.md"
)

# Track referenced sample files for embedding
$global:ReferencedSamples = @{}

# ============================================================================
# Helper Functions
# ============================================================================

function Get-MimeType {
    param([string]$Extension)
    switch ($Extension.ToLower()) {
        ".png" { return "image/png" }
        ".jpg" { return "image/jpeg" }
        ".jpeg" { return "image/jpeg" }
        ".gif" { return "image/gif" }
        ".webp" { return "image/webp" }
        ".svg" { return "image/svg+xml" }
        default { return "image/png" }
    }
}

function ConvertTo-Base64DataUri {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-Warning "Image not found: $FilePath"
        return $null
    }
    
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    $base64 = [Convert]::ToBase64String($bytes)
    $ext = [System.IO.Path]::GetExtension($FilePath)
    $mime = Get-MimeType $ext
    
    return "data:$mime;base64,$base64"
}

function Get-SampleAnchor {
    param([string]$FileName)
    
    # Create a URL-safe anchor from filename
    $anchor = $FileName -replace '\.md$', ''
    $anchor = $anchor -replace '_', '-'
    $anchor = $anchor.ToLower()
    return "sample-$anchor"
}

function Process-MarkdownLine {
    param([string]$Line)
    
    $processedLine = $Line
    
    # Process markdown image syntax: ![alt](../charts/image.png)
    if ($Line -match '!\[([^\]]*)\]\(([^)]+)\)') {
        $altText = $Matches[1]
        $imgPath = $Matches[2]
        
        if ($imgPath -match '^\.\./charts/') {
            $fullPath = $imgPath -replace '\.\./charts/', "$ChartsDir\"
            $fullPath = $fullPath -replace '/', '\'
            $dataUri = ConvertTo-Base64DataUri $fullPath
            
            if ($dataUri) {
                $processedLine = $Line -replace [regex]::Escape("![$altText]($imgPath)"), "![$altText]($dataUri)"
            }
        }
    }
    
    # Process HTML img tags: <img src="../charts/image.png" ...>
    if ($Line -match '<img\s+src="([^"]+)"') {
        $imgPath = $Matches[1]
        
        if ($imgPath -match '^\.\./charts/') {
            $fullPath = $imgPath -replace '\.\./charts/', "$ChartsDir\"
            $fullPath = $fullPath -replace '/', '\'
            $dataUri = ConvertTo-Base64DataUri $fullPath
            
            if ($dataUri) {
                $processedLine = $Line -replace [regex]::Escape($imgPath), $dataUri
            }
        }
    }
    
    # Process sample file links: [text](../samples/file.md)
    # Replace with anchor links and track for embedding
    $samplePattern = '\[([^\]]+)\]\(\.\./samples/([^)]+\.md)\)'
    if ($Line -match $samplePattern) {
        $matches = [regex]::Matches($Line, $samplePattern)
        foreach ($match in $matches) {
            $linkText = $match.Groups[1].Value
            $sampleFile = $match.Groups[2].Value
            $anchor = Get-SampleAnchor $sampleFile
            
            # Track this sample for embedding later
            $global:ReferencedSamples[$sampleFile] = $true
            
            # Replace with anchor link
            $originalLink = $match.Value
            $newLink = "[$linkText](#$anchor)"
            $processedLine = $processedLine -replace [regex]::Escape($originalLink), $newLink
        }
    }
    
    return $processedLine
}

function Process-SectionFile {
    param([string]$FilePath)
    
    $content = Get-Content $FilePath -Raw -Encoding UTF8
    $lines = $content -split "`r?`n"
    $processedLines = @()
    
    foreach ($line in $lines) {
        $processedLine = Process-MarkdownLine $line
        $processedLines += $processedLine
    }
    
    return $processedLines -join "`n"
}

# ============================================================================
# Main Processing
# ============================================================================

Write-Host "Adaptive-P Documentation Compiler" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$output = @()

# Header
$output += "# Adaptive-P Sampler - Complete Documentation"
$output += ""
$output += "> **Compiled on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')**"
$output += ">"
$output += "> This is a self-contained version of the Adaptive-P documentation with all images embedded as base64 and all sample files included inline. No external references."
$output += ""
$output += "---"
$output += ""

# Table of Contents
$output += "## Table of Contents"
$output += ""
$output += "- [Abstract](#abstract)"
$output += "- [1. Introduction](#1-introduction)"
$output += "- [2. Related Work and Comparative Analysis](#2-related-work-and-comparative-analysis)"
$output += "- [3. The Adaptive-P Algorithm](#3-the-adaptive-p-algorithm)"
$output += "- [5. Parameters](#5-parameters)"
$output += "- [6. Integration and Sampler Chain](#6-integration-and-sampler-chain)"
$output += "- [7. Empirical Validation](#7-empirical-validation)"
$output += "- [8. Implementation Reference](#8-implementation-reference)"
$output += "- [9. Conclusion](#9-conclusion)"
$output += "- [Appendix: Sample Outputs](#appendix-sample-outputs)"
$output += ""
$output += "---"
$output += ""

# Process each section
$totalSections = $SectionFiles.Count
$currentSection = 0

foreach ($sectionFile in $SectionFiles) {
    $currentSection++
    $sectionPath = Join-Path $SectionsDir $sectionFile
    
    Write-Host "[$currentSection/$totalSections] Processing $sectionFile..." -ForegroundColor Yellow
    
    if (Test-Path $sectionPath) {
        $content = Process-SectionFile $sectionPath
        $output += $content
        $output += ""
        $output += "---"
        $output += ""
    }
    else {
        Write-Warning "Section file not found: $sectionPath"
    }
}

# Embed all sample files as an appendix
Write-Host ""
Write-Host "Embedding sample files..." -ForegroundColor Yellow

$sampleFiles = Get-ChildItem -Path $SamplesDir -Filter "*.md" -ErrorAction SilentlyContinue

if ($sampleFiles) {
    $output += "## Appendix: Sample Outputs"
    $output += ""
    $output += "The following sample outputs are referenced throughout the documentation. Click on any sample link above to jump to that section."
    $output += ""
    
    foreach ($sampleFile in $sampleFiles) {
        Write-Host "  - Embedding $($sampleFile.Name)" -ForegroundColor Gray
        
        # Create anchor-friendly ID
        $anchor = Get-SampleAnchor $sampleFile.Name
        
        # Create readable title
        $sampleName = $sampleFile.BaseName -replace '_', ' '
        $sampleName = (Get-Culture).TextInfo.ToTitleCase($sampleName)
        
        $output += "<a id=`"$anchor`"></a>"
        $output += ""
        $output += "### $sampleName"
        $output += ""
        $content = Get-Content $sampleFile.FullName -Raw -Encoding UTF8
        $output += $content
        $output += ""
        $output += "---"
        $output += ""
    }
}

# Write output
Write-Host ""
Write-Host "Writing output to $OutputPath..." -ForegroundColor Green

$output -join "`n" | Out-File -FilePath $OutputPath -Encoding UTF8 -NoNewline

$fileSize = (Get-Item $OutputPath).Length / 1KB
Write-Host ""
Write-Host "Success! Documentation compiled." -ForegroundColor Green
Write-Host "  Output: $OutputPath" -ForegroundColor Cyan
Write-Host "  Size: $([math]::Round($fileSize, 2)) KB" -ForegroundColor Cyan
Write-Host "  Sample files embedded: $($sampleFiles.Count)" -ForegroundColor Cyan
Write-Host ""
