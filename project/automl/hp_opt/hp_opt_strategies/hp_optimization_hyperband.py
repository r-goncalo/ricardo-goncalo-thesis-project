

import os

from automl.component import InputSignature
from automl.hp_opt.hp_opt_strategies.hp_optimization_loader_detached import HyperparameterOptimizationPipelineLoaderDetached

import math
import optuna

 
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

MINIMUM_SLEEP = 1
SLEEP_INCR_RATE = 2
MAX_SLEEP = 60


class HyperparameterOptimizationPipelineHyperband(HyperparameterOptimizationPipelineLoaderDetached):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         "hyperband_eta": InputSignature(default_value=2),
                         "hyperband_min_steps": InputSignature(default_value=1),
                         "initial_number_of_trials" : InputSignature(mandatory=False),
                         "max_steps_per_trial" : InputSignature(mandatory=False)
                       }
    
    exposed_values = {"current_hyperband_bracket" : -1, "current_sucessive_halving_bracket" : -1} 
            
    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()

        self.eta = self.get_input_value("hyperband_eta")
        self.min_steps = self.get_input_value("hyperband_min_steps")
        self.max_steps = self.n_steps

        self.max_steps_per_trial = self.get_input_value("max_steps_per_trial")

        self.n_initial_hyperband_trials = self.get_input_value("initial_number_of_trials")

        self.lg.writeLine(f"Finished processing input related to hyperband execution")


    

    # RUNNING A TRIAL -----------------------------------------------------------------------
    
    
    def _reduce_trials_due_to_results(self, trials : list, results : list[tuple[optuna.Trial, int]], n_trials_to_mantain : int):

        if n_trials_to_mantain >= len(trials):
            return results

        results.sort(key=lambda x: x[1], reverse=(self.direction == "maximize"))

        trials_to_mantain : list[tuple[optuna.Trial, int]] = [(t, r) for t, r in results[:(n_trials_to_mantain + 1)]]

        if len(trials_to_mantain) > 0: # if there are still trials to mantain, we mark as pruned the trials that were discarded

            for trial_to_discard, result in results[(n_trials_to_mantain + 1):]:
                self.values["trials_done_in_this_execution"] += 1
                with self.optuna_usage_sem:
                    self.study.tell(trial=trial_to_discard, state= optuna.trial.TrialState.PRUNED) # this is prunning due to sucessive halving

                self.lg.writeLine(f"Marked trial {trial_to_discard.number} as pruned, with value {result}")

        else: # if no trials are mantained, then we mark all surviving trials as completed
            for trial, result in results:
                self.mark_trial_as_complete(trial, result)

        return trials_to_mantain

    def _resume_single_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate, i, total_step_budget):
            
            n_trials_to_mantain = len(trials)

            total_step_budget = 0  
            end_successive_halving = False

            step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i)

            stepd_that_should_be_done = total_step_budget + step_budget

            self.lg.writeLine(
                    f"Resuming: [HB] Bracket {number_of_runs} | Rung {i} | "
                    f"{len(trials)} trials | steps={step_budget} | total_steps={stepd_that_should_be_done}\n"
                )

            if self.max_steps_per_trial is not None and step_budget + total_step_budget > self.max_steps_per_trial:

                step_budget = self.max_steps_per_trial - total_step_budget
                end_successive_halving = True

                self.lg.writeLine(f"Because of limit of max steps per trial of {self.max_steps_per_trial} and current steps done {total_step_budget}, this will be last bracket with {step_budget} steps instead")

            results, completed_results, surviving_trials = self.run_trials(trials, running_method=self._try_resume_single_trial, steps_to_run=total_step_budget + step_budget, mark_trials_as_completed=False)

            n_trials_to_mantain = max(0, n_trials_to_mantain // configurations_to_study_decrease_rate) # the number of trials we expect to mantain (inclusive)
            trials_and_results_to_mantain : list[tuple[optuna.Trial, int]] = self._reduce_trials_due_to_results(surviving_trials, completed_results, n_trials_to_mantain)

            trials = [t for t, _ in trials_and_results_to_mantain]
            
            self.lg.writeLine(f"Step {i + 1} of successive halving: Expected to mantain {n_trials_to_mantain} from the original {len(results)} trials, mantained:")
            self.lg.writeLine(f"{[trial.number for trial in trials]},  from: {[trial.number for (trial, _) in results]}\n")

            end_successive_halving = end_successive_halving or len(trials) < 1

            return trials_and_results_to_mantain, trials, end_successive_halving
    
    def _resume_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate):

            total_step_budget = 0  
            end_successive_halving = False

            for i in range(self.values["current_sucessive_halving_bracket"]):
                step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i)
                total_step_budget += step_budget

            trials_and_results_to_mantain, trials, end_successive_halving = self._resume_single_successive_halving(trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate,
                                                                            self.values["current_sucessive_halving_bracket"], total_step_budget)

            total_step_budget += int(initial_resource_per_config * configurations_to_study_decrease_rate ** i)

            if not end_successive_halving:
                self._do_successive_halving(trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate, initial_i=self.values["current_sucessive_halving_bracket"] + 1)

            else:
                for trial, result in trials_and_results_to_mantain:
                    self.mark_trial_as_complete(trial, result)

            self.values["current_sucessive_halving_bracket"] = -1


    def _resume_true_hyperband_bracket(self, s, trials):

        r = int(self.max_steps * self.eta ** (-s))

        self._resume_successive_halving(trials, r, s, self.eta)


    def _resume_hyperband(self, trials):

        self.lg.writeLine(f"Resuming hyperband algorithm, last bracket registered was: {self.values['current_hyperband_bracket']}")

        s_max = int(math.log(self.max_steps / self.min_steps, self.eta))

        n_trials = self.n_initial_hyperband_trials

        if n_trials is not None:
            self.lg.writeLine(f"Initial number of trials passed of: {n_trials}")

        n_trials_done = 0

        # this is for simulating the hyperband bracket that were already done
        for s in reversed(range(s_max + 1)):

            if n_trials is None:
                n_trials = int(math.ceil((s_max + 1) / (s + 1) * self.eta ** s))

            n_trials_done += n_trials

            n_trials = None

            if s <= self.values["current_hyperband_bracket"]: # we iterate until we find the last s we were computing
                break

        self.lg.writeLine(f"Resuming last hyperband bracket that was being done: {s}, number of trials assumed to have been done by hyperband: {n_trials_done}\n")

        self._resume_true_hyperband_bracket(s, trials)

        self.lg.writeLine(f"Resuming normal hyperband execution...\n")

        n_trials_done += self._run_true_hyperband(s - 1)

        return n_trials_done
    

    def _try_resuming_unfinished_trials(self, trials : list[optuna.trial.FrozenTrial]):
        
        if self.values["current_hyperband_bracket"] > -1:
            self.lg.writeLine(f"Noticed hyperband was running before interruption of training process, resuming using it...")
            self._try_load_all_resumed_trials(trials)
            self._resume_hyperband(trials)

        elif self.values["current_sucessive_halving_bracket"] > -1:
            self.lg.writeLine(f"WARINING: Sucessive halving was still in effect while hyperband was over, ignoring this and using normal resuming strategy...")
            super()._try_resuming_unfinished_trials(trials)

        else:
            self.lg.writeLine(f"No hyperband execution noticed when resuming, using normal strategy...")
            super()._try_resuming_unfinished_trials(trials)
    

    def _do_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate, initial_i=0):
            
            n_trials_to_mantain = len(trials)

            total_step_budget = 0  
            end_successive_halving = False

            for i in range(initial_i, number_of_runs + 1):

                self.values["current_sucessive_halving_bracket"] = i

                step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i) # step budget is higher in later brackets, as we study less configurations
               
                self.lg.writeLine(
                    f"[HB] Bracket {number_of_runs} | Rung {i} | "
                    f"{len(trials)} trials | steps={step_budget}"
                )

                if self.max_steps_per_trial is not None and step_budget + total_step_budget > self.max_steps_per_trial:

                    step_budget = self.max_steps_per_trial - total_step_budget
                    end_successive_halving = True

                    self.lg.writeLine(f"Because of limit of max steps per trial of {self.max_steps_per_trial} and current steps done {total_step_budget}, this will be last bracket with {step_budget} steps instead")

                results, completed_results, surviving_trials = self.run_trials(trials, steps_to_run=step_budget, mark_trials_as_completed=False)

                if end_successive_halving:
                    trials_and_results_to_mantain = completed_results # this is so this trials are later marked as completed
                    break

                total_step_budget = total_step_budget + step_budget

                n_trials_to_mantain = max(0, n_trials_to_mantain // configurations_to_study_decrease_rate) # the number of trials we expect to mantain (inclusive)
                trials_and_results_to_mantain : list[tuple[optuna.Trial, int]] = self._reduce_trials_due_to_results(surviving_trials, completed_results, n_trials_to_mantain)

                trials = [t for t, _ in trials_and_results_to_mantain]

                self.lg.writeLine(f"Step {i + 1} of successive halving: Expected to mantain {n_trials_to_mantain} from the original {len(results)} trials, mantained:")
                self.lg.writeLine(f"{[trial.number for trial in trials]},  from: {[trial.number for (trial, _) in results]}\n")

                if len(trials) < 1:
                    break

            for trial, result in trials_and_results_to_mantain: # if there are still trials at the end of the successive halving, we mark them as done
                self.mark_trial_as_complete(trial, result)

            self.values["current_sucessive_halving_bracket"] = -1


  
    def _run_true_hyperband_bracket(self, s, n_trials):

        r = int(self.max_steps * self.eta ** (-s))

        with self.optuna_usage_sem:
            trials = [self.study.ask() for _ in range(n_trials)]

        self._do_successive_halving(trials, r, s, self.eta)


    def _run_true_hyperband(self, s_max=None):

        if s_max is None:
            s_max = int(math.log(self.max_steps / self.min_steps, self.eta))

        n_trials = self.n_initial_hyperband_trials

        if n_trials is not None:
            self.lg.writeLine(f"Initial number of trials passed of: {n_trials}")

        n_trials_done = 0

        for s in reversed(range(s_max + 1)):

            self.values["current_hyperband_bracket"] = s

            end_early = False

            if n_trials is None:
                n_trials = int(math.ceil((s_max + 1) / (s + 1) * self.eta ** s))

            if n_trials > self.n_trials:
                self.lg.writeLine(f"Trials to do in bracket ({n_trials}) is higher than trials that are yet to be done: {self.n_trials}, adapting it")
                n_trials = self.n_trials
                end_early = True

            self._run_true_hyperband_bracket(s, n_trials)

            n_trials_done += n_trials

            n_trials = None

            if end_early:
                break

        self.values["current_hyperband_bracket"] = -1

        return n_trials_done

    

  
    def _call_objective(self):

        self.n_trials -= self._run_true_hyperband()

        while self.n_trials >  0:
            self.lg.writeLine(f"Hyperband did not complete asked number of trials, {self.n_trials} missing")

            trials_done = self._run_true_hyperband()
            self.n_trials -= trials_done



        


