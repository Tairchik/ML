"""Update the DOCX fields and render PDF with installed Windows Word from WSL."""
import subprocess
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    # Render a private copy so an open report in Word does not block automation.
    with tempfile.TemporaryDirectory(prefix='word-render-', dir=runtime) as folder:
        folder = Path(folder)
        working_docx, working_pdf = folder / 'report.docx', folder / 'report.pdf'
        shutil.copyfile(ROOT / 'Отчёт_ЛР4_MLOps.docx', working_docx)
        paths = [ROOT / 'scripts/export_word_pdf.ps1', working_docx, working_pdf]
        windows_paths = [subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
                         for path in paths]
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', windows_paths[0],
            '-InputPath', windows_paths[1], '-OutputPath', windows_paths[2]],
            capture_output=True, timeout=180)
        print(result.stdout.decode('utf-8', errors='replace'))
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
        working_docx.replace(ROOT / 'Отчёт_ЛР4_MLOps.docx')
        working_pdf.replace(ROOT / 'Отчёт_ЛР4_MLOps.pdf')


if __name__ == '__main__':
    main()
