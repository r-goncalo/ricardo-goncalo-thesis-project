@echo on

REM === Set logs directory ===
set LOGDIR=C:\rgoncalo\logs

REM Create logs directory if it doesn't exist
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set LOGBASENAME=batches_log

REM === Set timestamp for log file ===
for /f %%A in ('powershell -NoProfile -Command "Get-Date -Format yyyy_MM_dd_HH_mm_ss"') do set TIMESTAMP=%%A

REM === Set log file path with timestamp ===
set LOGFILE=%LOGDIR%\%LOGBASENAME%_%TIMESTAMP%.txt

echo Starting at %TIME% >> %LOGFILE%

CALL C:/rgoncalo/ricardo-goncalo-thesis-project/.conda/python.exe C:\rgoncalo\ricardo-goncalo-thesis-project\other_code\run_multiple_experiments_windows.py  >> %LOGFILE% 2>&1

echo Finished at %TIME% >> %LOGFILE%



