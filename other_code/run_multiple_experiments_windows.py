import subprocess, win32job, win32process, win32con, win32api
import multiprocessing
import experiments

# === Define Maximum Concurrent Jobs ===
MAX_JOBS = 2



# --- Create a global Job Object ---
job = win32job.CreateJobObject(None, "")

# Configure it so all processes in the job are killed when the job handle closes
extended_info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
extended_info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, extended_info)


# Function to execute a command
def run_job(command):
    if isinstance(command, str):
        print(f"Starting string job: {command}", flush=True)
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    elif isinstance(command, list):
        print(f"Starting job with args: {command}", flush=True)
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
def run_jobs_concurrently(commands):

    print(f"Max jobs value is: {MAX_JOBS}")

    # Create a pool of processes
    with multiprocessing.Pool(processes=MAX_JOBS) as pool:
        # Use map to apply the run_job function to each command in the list
        results = pool.map(run_job, commands)
    
    print("All jobs have finished.")

# === Run the jobs ===
if __name__ == "__main__":
    print("Running multiple experiments on windows as main", flush=True)


    #commands = experiments.hp_opt_for_models(directory_of_models="C:\\rgoncalo\experiment_definitions\\dqn_montaincar_sb3_zoo_semi_trained_2\\models", 
    #                             directory_to_store_experiment="C:\\rgoncalo\\experiments",
    #                             base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_montaincar_sb3_zoo_semi_trained_2\\to_optimize_configuration.json", 
    #                             hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_montaincar_sb3_zoo_semi_trained_2\\configuration.json")

    commands = experiments.hp_opt_for_models(
                            directory_of_models="C:\\rgoncalo\experiment_definitions\\dqn_cartpole_sb3_zoo\\models", 
                            directory_to_store_experiment="C:\\rgoncalo\\experiments",
                            base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json", 
                            hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration.json",
                            experiment_name="sb3_zoo_dqn_cartpole_hp_opt"
                            )



    run_jobs_concurrently(commands)
