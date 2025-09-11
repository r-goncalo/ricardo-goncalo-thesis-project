import subprocess
import multiprocessing

# === Define Maximum Concurrent Jobs ===
MAX_JOBS = 3

SCRIPT_PATH = "C:\\rgoncalo\\RunSimplyHpExp.bat"

# === Define the list of commands to run ===
commands = [
    [SCRIPT_PATH],
    [SCRIPT_PATH, "--LOGBASENAME", "sb3_montaincar_semi_trained_reduced_75",
     "--EXPSTOREPATH", "sb3_montaincar_semi_trained_reduced_75",
     "--TOOPTIMIZECONFIG", "to_optimize_configuration_75.json",
     "--CONFIG", "configuration_reduced.json"],
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_100 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_100 --TOOPTIMIZECONFIG to_optimize_configuration_100.json --CONFIG configuration_reduced.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_150 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_150 --TOOPTIMIZECONFIG to_optimize_configuration_150.json --CONFIG configuration_reduced.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_200 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_200 --TOOPTIMIZECONFIG to_optimize_configuration_200.json --CONFIG configuration_reduced.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_75  --EXPSTOREPATH sb3_montaincar_semi_trained_75 --TOOPTIMIZECONFIG to_optimize_configuration_75.json --CONFIG configuration.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_100 --EXPSTOREPATH sb3_montaincar_semi_trained_100 --TOOPTIMIZECONFIG to_optimize_configuration_100.json --CONFIG configuration.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_150 --EXPSTOREPATH sb3_montaincar_semi_trained_150 --TOOPTIMIZECONFIG to_optimize_configuration_150.json --CONFIG configuration.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_200 --EXPSTOREPATH sb3_montaincar_semi_trained_200 --TOOPTIMIZECONFIG to_optimize_configuration_200.json --CONFIG configuration.json',
]

# == Windows Stuff

job = win32job.CreateJobObject(None, "")
extended_info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
extended_info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, extended_info)

# Function to execute a command
def run_job(command):
    if isinstance(command, str):
        print(f"Starting string job: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    elif isinstance(command, list):
        print(f"Starting job with args: {command}")
        result = subprocess.run(command, capture_output=True, text=True)

    else:
        print("WARNING: JOB WILL BE IGNORED, NOT IN SUPPORTED FORMAT")
    
    if result.returncode == 0:
        print(f"Job completed successfully: {command}")
    else:
        print(f"Job failed: {command}\nError: {result.stderr}")
    return result.returncode

# === Using multiprocessing to run jobs concurrently ===
def run_jobs_concurrently():
    # Create a pool of processes
    with multiprocessing.Pool(processes=MAX_JOBS) as pool:
        # Use map to apply the run_job function to each command in the list
        results = pool.map(run_job, commands)
    
    print("All jobs have finished.")

# === Run the jobs ===
if __name__ == "__main__":
    run_jobs_concurrently()
