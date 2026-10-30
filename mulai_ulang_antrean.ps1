# =====================================================================================
# mulai_ulang_antrean.ps1 - Menjalankan ulang antrean training v9 (prov_v9 latar dihapus: 6 run final)
#
# CARA MEMAKAI (3 langkah):
#   1. Buka PowerShell, lalu masuk ke folder project:
#        cd D:\TELKOM\SKRIPSI\rice_yolo
#   2. Jalankan berkas ini:
#        powershell -ExecutionPolicy Bypass -File .\mulai_ulang_antrean.ps1
#   3. Cek status (boleh diulang kapan saja):
#        .\.venv\Scripts\python.exe scripts\status_final.py
#
# Yang dilakukan berkas ini:
#   - Antrean yang SUDAH berjalan tidak dijalankan lagi (tidak dobel).
#   - Satu antrean (scripts/antrean_v9.py): 6 run final (YOLOv8s dan YOLO11s x seed 0, 1, 2), lalu kunci ambang di VAL v9.
#     (Antrean v7 - antrean_v7.py - dan v6 sudah selesai dan tidak dijalankan lagi.)
#   - Run yang sudah selesai dilewati; run yang terputus dilanjutkan (resume) oleh antreannya.
#   - Bila ada berkas BERHENTI (permintaan berhenti dari Anda), antrean itu TIDAK dijalankan.
#     Hapus dulu berkas runs\_antrean_v9\BERHENTI bila memang ingin melanjutkan.
#   - Log keluaran tiap peluncuran disimpan dengan cap waktu (log lama tidak tertimpa).
# =====================================================================================

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Waktu = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $Py)) {
    Write-Host "[GAGAL] python venv tidak ditemukan: $Py"
    exit 1
}

function Cari-Proses([string]$Skrip) {
    # Proses python yang baris perintahnya memuat nama skrip antrean (venv bisa memunculkan 2 proses: peluncur + interpreter)
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Skrip*" }
}

function Mulai-Antrean([string]$Nama, [string]$Skrip, [string]$Folder) {
    $dir = Join-Path $Root $Folder
    New-Item -ItemType Directory -Force $dir | Out-Null
    $jalan = @(Cari-Proses $Skrip)
    if ($jalan.Count -gt 0) {
        Write-Host ("[LEWATI] {0} sudah berjalan (PID {1}); tidak dijalankan lagi." -f $Nama, (($jalan | ForEach-Object { $_.ProcessId }) -join ", "))
        return
    }
    if (Test-Path (Join-Path $dir "BERHENTI")) {
        Write-Host ("[LEWATI] {0}: ada berkas {1}\BERHENTI (permintaan berhenti). Hapus berkas itu bila ingin melanjutkan." -f $Nama, $Folder)
        return
    }
    $out = Join-Path $dir ("stdout_{0}.txt" -f $Waktu)
    $err = Join-Path $dir ("stderr_{0}.txt" -f $Waktu)
    $p = Start-Process -FilePath $Py -ArgumentList ("scripts\" + $Skrip) -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Start-Sleep -Seconds 10
    $jalan = @(Cari-Proses $Skrip)
    if ($jalan.Count -gt 0) {
        Write-Host ("[MULAI] {0} berjalan (PID {1}). Log: {2}\antrean.log" -f $Nama, (($jalan | ForEach-Object { $_.ProcessId }) -join ", "), $Folder)
    } else {
        Write-Host ("[GAGAL] {0} berhenti segera setelah dijalankan. Lihat {1}" -f $Nama, $err)
        if (Test-Path $err) { Get-Content $err -Tail 5 }
    }
}

Write-Host "Folder project: $Root"
# antrean v9 (cek memori, resume otomatis, dan berkas BERHENTI ditangani oleh antrean itu sendiri)
Mulai-Antrean "Antrean v9" "antrean_v9.py" "runs\_antrean_v9"

Write-Host ""
Write-Host "Cek status: .\.venv\Scripts\python.exe scripts\status_final.py"
