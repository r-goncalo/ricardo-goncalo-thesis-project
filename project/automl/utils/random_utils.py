
import random
import numpy
import torch


def generate_and_setup_a_seed():
    
    seed = generate_seed()

    do_full_setup_of_seed(seed)
    
    return seed



def generate_seed():
    return random.randint(0, 2**32 - 1)

    
def do_full_setup_of_seed(seed):
    
    setup_torch_seed(seed)
    setup_numpy_seed(seed)
    setup_python_seed(seed)

    global full_setup_of_seed_was_done
    full_setup_of_seed_was_done = True


def setup_python_seed(seed):
    random.seed(seed)    


def setup_numpy_seed(seed):
    numpy.random.seed(seed)     # NumPy    


def setup_torch_seed(seed):
    
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False