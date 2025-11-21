
import random
import numpy
from automl.loggers.global_logger import globalWriteLine
from automl.loggers.logger_component import DEBUG_LEVEL
import torch

SEED_GLOBAL_LOGGER = "seed_logging.txt"


def generate_and_setup_a_seed():
    
    seed = generate_seed()

    do_full_setup_of_single_seed(seed)
    
    return seed



def generate_seed():
    return random.randint(0, 2**32 - 1)


def generate_seed_configuration(seed_configuration : dict = None):

    seed_configuration = {} if seed_configuration == None else {**seed_configuration}

    if seed_configuration.get("python", None) == None:
        seed_configuration["python"] = generate_seed()

    if seed_configuration.get("torch", None) == None:
        seed_configuration["torch"] = generate_seed()

    if seed_configuration.get("numpy", None) == None:
        seed_configuration["numpy"] = generate_seed()

    return seed_configuration


def setup_seed_from_dict_configuration(seed_configuration : dict):

    seed_configuration = generate_seed_configuration(seed_configuration)

    setup_python_seed(seed_configuration["python"])

    setup_torch_seed(seed_configuration["torch"])

    setup_numpy_seed(seed_configuration["numpy"])
                     
    return seed_configuration
    

def do_full_setup_of_single_seed(seed):

    setup_python_seed(seed)
    
    return setup_seed_from_dict_configuration(
        {
            "python" : seed
        }
    )


def setup_python_seed(seed):
    globalWriteLine(f"Python seed {seed}", file=SEED_GLOBAL_LOGGER)

    random.seed(seed)    


def setup_numpy_seed(seed):
    globalWriteLine(f"Numpy seed {seed}", file=SEED_GLOBAL_LOGGER)
    numpy.random.seed(seed)     # NumPy    


def setup_torch_seed(seed):

    globalWriteLine(f"Torch seed {seed}", file=SEED_GLOBAL_LOGGER)
    
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False