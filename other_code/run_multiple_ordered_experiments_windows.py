import subprocess, win32job, win32process, win32con, win32api
import multiprocessing
import experiments_sequence

# === Define Maximum Concurrent Jobs ===
MAX_JOBS = 2

# --- Create a global Job Object ---
job = win32job.CreateJobObject(None, "")

# Configure it so all processes in the job are killed when the job handle closes
extended_info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
extended_info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, extended_info)


# Function to execute a single command
def run_command(command):

    if isinstance(command, str):
        print(f"Starting string job: {command}", flush=True)
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    elif isinstance(command, list):
        print(f"Starting job with args: {command}", flush=True)
        proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    else:
        print("WARNING: JOB WILL BE IGNORED, NOT IN SUPPORTED FORMAT")
        return 1  # error return code

    # Assign subprocess to Job Object
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, proc.pid)
    win32job.AssignProcessToJobObject(job, handle)

    stdout, stderr = proc.communicate()

    if proc.returncode == 0:
        print(f"Job completed successfully: {command}")
    else:
        print(f"Job failed: {command}\nError: {stderr}")

    return proc.returncode


# Function to execute a sequence of commands (synchronously, in order)

def run_command_sequence(command_list):
        
    for cmd in command_list:
        ret = run_command(cmd)
        if ret != 0:  # Stop if one fails
            print(f"Aborting sequence due to failure in: {cmd}", flush=True)
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

    with multiprocessing.Pool(processes=MAX_JOBS) as pool:
        results = pool.map(run_command_sequence, command_groups)

    print("All jobs have finished.")
    return results


# === Run the jobs ===
if __name__ == "__main__":
    print("Running multiple experiments on windows as main", flush=True)

    # Example: now experiments.hp_opt_for_models should return list of lists
    # [
    #   ["python train.py --model m1", "python eval.py --model m1"],
    #   ["python train.py --model m2", "python eval.py --model m2"]
    # ]
    command_sequences = experiments_sequence.hp_opt_for_models(
        directory_of_models="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models", 
        directory_to_store_experiment="C:\\rgoncalo\\experiments",
        base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json", 
        hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration.json",
        experiment_name="sb3_zoo_dqn_cartpole_hp_opt_mult_samplers_pruners",

        parameter_dict_list=[
            {
                "num_trials" : 20,
                "sampler" : "Random", # this is to gain some knowledge first
                
            },
            {
                "num_trials" : 80,
                "sampler": "TreeParzen"
            }
        ],

        models_to_test= ["sb3_CartPole_dqn", "sb3_CartPole_dqn_perturbed_0_10"]
    )

    print("\nCommand_sequences: ")

    for command_sequence in command_sequences:
        print("\n    Sequence: \n")
        for command in command_sequence:
            print("        " + str(command_sequence))
        print("\nEnd of sequence\n")

    print()

    run_jobs_concurrently(command_sequences)
