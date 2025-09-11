@echo on

REM === Default Values ===
set LOGDIR=C:\rgoncalo\logs
set LOGBASENAME=experiment_log
set BASEEXPTOSTORE=C:\rgoncalo\experiments
set EXPDEF=C:\rgoncalo\experiment_definitions\dqn_montaincar_sb3_zoo_semi_trained
set RELPATH=Experiment
set EXPSTOREPATH=sb3_montaincar_semi_trained_reduced_50
set TOOPTIMIZECONFIG=to_optimize_configuration_50.json
set CONFIG=configuration_reduced.json

REM === Parse Command-Line Arguments ===
:parse_args
if "%~1"=="" goto done
if "%~1"=="--LOGDIR" (
    set LOGDIR=%~2
    shift
) else if "%~1"=="--LOGBASENAME" (
    set LOGBASENAME=%~2
    shift
) else if "%~1"=="--BASEEXPTOSTORE" (
    set BASEEXPTOSTORE=%~2
    shift
) else if "%~1"=="--EXPDEF" (
    set EXPDEF=%~2
    shift
) else if "%~1"=="--RELPATH" (
    set RELPATH=%~2
    shift
) else if "%~1"=="--EXPSTOREPATH" (
    set EXPSTOREPATH=%~2
    shift
) else if "%~1"=="--TOOPTIMIZECONFIG" (
    set TOOOPTIMIZECONFIG= %~2
    shift
)
shift
goto parse_args

:done

REM === Create logs directory if it doesn't exist ===
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM === Set timestamp for log file ===
for /f %%A in ('powershell -NoProfile -Command "Get-Date -Format yyyy_MM_dd_HH_mm_ss"') do set TIMESTAMP=%%A

REM === Set log file path with timestamp ===
set LOGFILE=%LOGDIR%\%LOGBASENAME%_%TIMESTAMP%.txt

REM === Activate Conda environment ===
CALL "C:\Users\ricar\anaconda3\Scripts\activate.bat" "C:\Users\ricar\anaconda3\envs\rl"

REM Guarantee module is installed
CALL pip install -e C:\rgoncalo\ricardo-goncalo-thesis-project\project

echo Starting experiment at %TIME% >> %LOGFILE%

python C:\rgoncalo\ricardo-goncalo-thesis-project\project\examples\simple_metarl\scripts\run_hp_experiment.py --path_to_store_experiment %BASEEXPTOSTORE%\%EXPSTOREPATH% --hp_configuration_path %EXPDEF%\%CONFIG% --to_optimize_configuration_path %EXPDEF%\%TOOPTIMIZECONFIG% --experiment_relative_path %RELPATH% >> %LOGFILE% 2>&1

echo Finished at %TIME% >> %LOGFILE%
