import subprocess, win32job, win32process, win32con, win32api
import multiprocessing

# === Define Maximum Concurrent Jobs ===
MAX_JOBS = 3

SCRIPT_PATH = "C:\\rgoncalo\\ricardo-goncalo-thesis-project\\other_code\\RunSimplyHpExp.bat"

# === Define the list of commands to run ===
commands = [

    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_10 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_10 --TOOPTIMIZECONFIG to_optimize_configuration_10.json --CONFIG configuration_reduced.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_30 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_30 --TOOPTIMIZECONFIG to_optimize_configuration_30.json --CONFIG configuration_reduced.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_50 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_50 --TOOPTIMIZECONFIG to_optimize_configuration_50.json --CONFIG configuration_reduced.json',
    
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_10 --EXPSTOREPATH sb3_montaincar_semi_trained_10 --TOOPTIMIZECONFIG to_optimize_configuration_10.json --CONFIG configuration.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_30 --EXPSTOREPATH sb3_montaincar_semi_trained_30 --TOOPTIMIZECONFIG to_optimize_configuration_30.json --CONFIG configuration.json',
    f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_50 --EXPSTOREPATH sb3_montaincar_semi_trained_50 --TOOPTIMIZECONFIG to_optimize_configuration_50.json --CONFIG configuration.json',

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

# --- Create a global Job Object ---
job = win32job.CreateJobObject(None, "")

# Configure it so all processes in the job are killed when the job handle closes
extended_info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
extended_info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, extended_info)


# Function to execute a command
def run_job(command):
    if isinstance(command, str):
        print(f"Starting string job: {command}")
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    elif isinstance(command, list):
        print(f"Starting job with args: {command}")
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    else:
        print("WARNING: JOB WILL BE IGNORED, NOT IN SUPPORTED FORMAT")
    
    # Assign this subprocess to the Job Object
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, proc.pid)
    win32job.AssignProcessToJobObject(job, handle)

    # Wait for it (if you want to collect output here)
    stdout, stderr = proc.communicate()
    
    if proc.returncode == 0:
        print(f"Job completed successfully: {command}")
    else:
        print(f"Job failed: {command}\nError: {stderr}")
    
    return proc.returncode

# === Using multiprocessing to run jobs concurrently ===
def run_jobs_concurrently():

    print(f"Max jobs value is: {MAX_JOBS}")

    # Create a pool of processes
    with multiprocessing.Pool(processes=MAX_JOBS) as pool:
        # Use map to apply the run_job function to each command in the list
        results = pool.map(run_job, commands)
    
    print("All jobs have finished.")

# === Run the jobs ===
if __name__ == "__main__":
    print("Running multiple experiments on windows as main")
    run_jobs_concurrently()
