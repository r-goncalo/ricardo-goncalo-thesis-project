import optuna
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from typing import Any, Dict, Optional


class AlternatingRandomSampler(BaseSampler):

    def __init__(
        self,
        sampler : BaseSampler = None
    ):
        # Default sampler: TPE with 5 random startup trials
        if sampler is None:
            sampler = TPESampler(n_startup_trials=5)

        self._base_sampler = sampler
        self._random_sampler = RandomSampler()
        self._use_random = True  # start with random


    def reseed_rng(self) -> None:
        self._base_sampler.reseed_rng()
        self._random_sampler.reseed_rng()


    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:

        if self._use_random:
            return self._random_sampler.infer_relative_search_space(study, trial)
        else:
            return self._base_sampler.infer_relative_search_space(study, trial)


    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution]
    ) -> Dict[str, Any]:

        if self._use_random:
            return self._random_sampler.sample_relative(study, trial, search_space)
        else:
            return self._base_sampler.sample_relative(study, trial, search_space)


    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:

        if self._use_random:
            value = self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            value = self._base_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        return value


    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[list[float]] = None,
    ) -> None:

        # Forward callback to both samplers
        self._random_sampler.after_trial(study, trial, state, values)
        self._base_sampler.after_trial(study, trial, state, values)

        # Alternate sampler for next trial
        self._use_random = not self._use_random