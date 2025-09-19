from automl.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn

from automl.component import InputSignature, requires_input_proccess
from automl.ml.models.model_components import ModelComponent
    
    
def perturb_model_parameters(
    torch_model: TorchModelComponent,
    min_percentage: float,
    max_percentage: float
):
    """
    Randomly perturbs model parameters within a given percentage range.

    @param min_percentage: float, e.g., 0.05 means perturbations will be at least ±5%.
    @param max_percentage: float, e.g., 0.1 means perturbations will be at most ±10%.
                           Each parameter will be multiplied by a random factor in
                           [0.9, 0.95] ∪ [1.05, 1.1] if min=0.05 and max=0.1.
    """

    if min_percentage < 0 or max_percentage < 0:
        raise ValueError("Percentages must be non-negative")
    if min_percentage > max_percentage:
        raise ValueError("min_percentage cannot be greater than max_percentage")

    if max_percentage == 0:
        return  # nothing to do

    if "perturbed_percentage" in torch_model.values.keys():
        print("WARNING: model has already had its parameters perturbed")

    torch_model.proccess_input_if_not_proccesd()

    with torch.no_grad():
        for param in torch_model.model.parameters():
            # Generate random multipliers that enforce min_percentage
            random_sign = torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
            random_magnitude = torch.empty_like(param).uniform_(min_percentage, max_percentage)
            factor = 1.0 + random_sign * random_magnitude
            param.mul_(factor)



def perturb_model_parameters_gaussian(
    torch_model: TorchModelComponent,
    mean: float = 0.0,
    std: float = 0.1,
    fraction: float = 1.0
):
    """
    Perturbs model parameters by adding Gaussian noise.

    @param mean: Mean of the Gaussian noise (default = 0.0).
    @param std: Standard deviation of the Gaussian noise (default = 0.1).
    @param fraction: Fraction of parameters to perturb (default = 1.0, i.e. all).
                     For example, 0.1 perturbs 10% of parameters.
    """

    if std < 0:
        raise ValueError("Standard deviation must be non-negative")
    if not (0 < fraction <= 1.0):
        raise ValueError("Fraction must be in (0,1]")

    torch_model.proccess_input_if_not_proccesd()

    with torch.no_grad():
        for param in torch_model.model.parameters():
            if torch.rand(1).item() < fraction:
                noise = torch.randn_like(param) * std + mean
                param.add_(noise)



def perturb_model_parameters_partial_forgetting(
    torch_model: TorchModelComponent,
    fraction: float = 0.1,
    std: float = 0.1
):
    """
    Perturbs model parameters by randomly reinitializing a fraction of them.

    @param fraction: Fraction of parameters to 'forget' (default = 0.1, i.e. 10%).
    @param std: Standard deviation of the reinitialized values (default = 0.1).
    """

    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    if std <= 0:
        raise ValueError("std must be positive.")

    if "perturbed_partial_forgetting" in torch_model.values.keys():
        print("WARNING: model has already had partial forgetting applied")

    torch_model.proccess_input_if_not_proccesd()

    with torch.no_grad():
        for param in torch_model.model.parameters():
            mask = torch.rand_like(param) < fraction
            param[mask] = torch.randn_like(param[mask]) * std


def model_parameter_distance(model_a : TorchModelComponent, model_b : TorchModelComponent):
    """Compute L2 distance and cosine similarity between two models."""
    params_a = torch.cat([p.flatten() for p in model_a.model.parameters()])
    params_b = torch.cat([p.flatten() for p in model_b.model.parameters()])
    
    l2_distance = torch.norm(params_a - params_b, p=2).item()
    avg_distance = l2_distance / params_a.numel()
    cosine_sim = torch.nn.functional.cosine_similarity(
        params_a.unsqueeze(0), params_b.unsqueeze(0)
    ).item()
    return l2_distance, avg_distance, cosine_sim


def model_output_difference(model_a : TorchModelComponent, model_b : TorchModelComponent, inputs):

    with torch.no_grad():
        out_a = model_a.model(inputs)
        out_b = model_b.model(inputs)
    mse = torch.nn.functional.mse_loss(out_a, out_b).item()
    return mse