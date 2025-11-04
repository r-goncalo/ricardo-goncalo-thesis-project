from datetime import datetime
import subprocess, win32job, win32process, win32con, win32api
import multiprocessing

# === Define Maximum Concurrent Jobs ===
MAX_JOBS = 4

# --- Create a global Job Object ---
job = win32job.CreateJobObject(None, "")

# Configure it so all processes in the job are killed when the job handle closes
extended_info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
extended_info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, extended_info)


# Function to execute a single command
def run_command(command):

    if isinstance(command, str):
        print(f"\nStarting string job: {command}\n", flush=True)
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    elif isinstance(command, list):
        print(f"\nStarting job with args: {command}\n", flush=True)
        proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    else:
        print(f"\nWARNING: JOB WITH COMMAND <{str(command)[:20]}...> WILL BE IGNORED, NOT IN SUPPORTED FORMAT")
        return 1  # error return code

    # Assign subprocess to Job Object
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, proc.pid)
    win32job.AssignProcessToJobObject(job, handle)

    stdout, stderr = proc.communicate()

    if proc.returncode == 0:
        print(f"\nJob completed successfully: <{str(command)[:20]}...>\n")
    else:
        print(f"\nJob failed: <{str(command)[:20]}...>\nError: {stderr}\n")

    return proc.returncode


# Function to execute a sequence of commands (synchronously, in order)

def run_command_sequence(command_list):
        
    for cmd in command_list:
        ret = run_command(cmd)
        if ret != 0:  # Stop if one fails
            print(f"Aborting sequence due to failure in: <{str(cmd)[:20]}...>", flush=True)
            return ret
    print(f"Finished sequence successfully: {command_list}", flush=True)
    return 0


# === Using multiprocessing to run multiple sequences concurrently ===
def run_jobs_concurrently(command_groups):
    """
    command_groups = [
        [cmd1, cmd2, cmd3],   # executed sequentially in one process
        [cmdA, cmdB],         # executed sequentially in another process
        ...
    ]
    """
    print(f"Max jobs value is: {MAX_JOBS}")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with multiprocessing.Pool(processes=MAX_JOBS) as pool:
        results = pool.map(run_command_sequence, command_groups)

    print(f"All jobs have finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    return results


# === Run the jobs ===
if __name__ == "__main__":

    print("Running multiple experiments on windows as main", flush=True)

    # Example: now experiments.hp_opt_for_models should return list of lists
    # [
    #   ["python train.py --model m1", "python eval.py --model m1"],
    #   ["python train.py --model m2", "python eval.py --model m2"]
    # ]

    from experiments.hp_experiments_sequence import print_commands, make_command_dicts_command_strings, unfold_sequences_to_correct_format
    from experiments.rl_zoo_sb3.ppo_cartpole import experiment_for_poo_actors_and_critics as experiment
    
    command_sequences = experiment(
        directory_to_store_experiment='C:\\rgoncalo\\experiments',
        base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\ppo_cartpole_sb3_zoo\\definitions\\to_optimize.json",
        hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\ppo_cartpole_sb3_zoo\\definitions\\configuration.json", 
        directory_of_models="C:\\rgoncalo\\experiment_definitions\\ppo_cartpole_sb3_zoo\\models\\actor_models",
        directory_of_critics="C:\\rgoncalo\\experiment_definitions\\ppo_cartpole_sb3_zoo\\models\\critic_models",
        mantain_critic_original=True,
        experiment_name="sb3_zoo_ppo_cartpole_hp_opt",
    
    )

    command_sequences = make_command_dicts_command_strings(command_sequences)

    command_sequences = unfold_sequences_to_correct_format(command_sequences)


    print("EXPERIMENTS TO DO:")

    print_commands(command_sequences)

    print("\nSTARTING THREADS TO DO EXPERIMENTS", flush=True)

    run_jobs_concurrently(command_sequences)

    