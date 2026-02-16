from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time
from automl.basic_components.component_group import RunnableComponentGroup, setup_component_group

from automl.component import InputSignature, Component
from automl.core.exceptions import common_exception_handling
from automl.hp_opt.hp_opt_strategies.hp_optimization_loader_detached import HyperparameterOptimizationPipelineLoaderDetached
from automl.hp_opt.hp_opt_strategies.workers.hp_worker import HyperparameterOptimizationWorkerIndexed
from automl.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import math

from automl.rl.evaluators.rl_std_avg_evaluator import ResultLogger
import optuna

 
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from automl.loggers.logger_component import use_logger

import pandas

MINIMUM_SLEEP = 1
SLEEP_INCR_RATE = 2
MAX_SLEEP = 60


OPTUNA_STUDY_PATH = "journal.log"

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
            


    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
    

    # RUNNING A TRIAL -----------------------------------------------------------------------
    


    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''
        Called when a trial is over
        It is passed to optuna in the callbacks when the objective is defined
        '''        

        super().after_trial(study, trial)
                        
        component_group = self._get_loader_component_group(trial)
        component_group.unload_all_components()

    
    def _initialize_database(self):
        
        self.database_path = os.path.join(self.get_artifact_directory(), OPTUNA_STUDY_PATH)  # Path to the SQLite database file
        
        self.lg.writeLine(f"Trying to initialize database in path: {self.database_path}")

        self.storage = JournalStorage(JournalFileBackend(file_path=self.database_path))

    
    def _run_single_trial(self, trial):

        was_pruned = False

        try:
            value = self.objective(trial)

        except optuna.TrialPruned:
            was_pruned = True

        return trial, value, was_pruned


    def _do_successive_halving_bracket(self, trials, step_budget):
        

        results = []
        executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)
        futures = []

        completed_results = []
        surviving_trials = []

        old_n_steps = self.n_steps
        self.n_steps = step_budget

        self.step_budget = step_budget

        for trial in trials:

            futures.append(
                executor.submit(self._run_single_trial, trial)
            )

        for future in as_completed(futures):
            trial, value, was_pruned = future.result()
            
            if was_pruned:
                self.study.tell(trial=trial, state=optuna.trial.TrialState.PRUNED)
            else:
                completed_results.append((trial, value))
                surviving_trials.append(trial)

            results.append((trial, value))

        executor.shutdown(wait=True)

        self.n_steps = old_n_steps


        return results, completed_results, surviving_trials

    
    
    
    def _reduce_trials_due_to_results(self, trials : list, results : list, n_trials_to_mantain : int):

        if n_trials_to_mantain >= len(trials):
            return trials

        results.sort(key=lambda x: x[1], reverse=(self.direction == "maximize"))

        trials_to_mantain = [t for t, _ in results[:(n_trials_to_mantain + 1)]]

        if len(trials_to_mantain) > 0: # if there are still trials to mantain, we mark as pruned the trials that were discarded

            for trial_to_discard, result in results[(n_trials_to_mantain + 1):]:
                self.study.tell(trial=trial_to_discard, state= optuna.trial.TrialState.PRUNED)

        else: # if no trials are mantained, then we mark all surviving trials as completed
            for trial, result in results:
                self.study.tell(trial, result)

        return trials_to_mantain
    

    def _do_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate):
            
            n_trials_to_mantain = len(trials)

            total_step_budget = 0  
            end_successive_halving = False

            for i in range(number_of_runs + 1):

                step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i) # step budget is higher in later brackets, as we study less configurations
               
                self.lg.writeLine(
                    f"[HB] Bracket {number_of_runs} | Rung {i} | "
                    f"{len(trials)} trials | steps={step_budget}"
                )


                if self.max_steps_per_trial is not None and step_budget + total_step_budget > self.max_steps_per_trial:

                    step_budget = self.max_steps_per_trial - total_step_budget
                    end_successive_halving = True

                    self.lg.writeLine(f"Because of limit of max steps per trial of {self.max_steps_per_trial} and current steps done {total_step_budget}, this will be last bracket with {step_budget} steps instead")

                results, completed_results, surviving_trials = self._do_successive_halving_bracket(trials, step_budget)
                
                if end_successive_halving:
                    break

                total_step_budget = total_step_budget + step_budget

                n_trials_to_mantain = max(0, n_trials_to_mantain // configurations_to_study_decrease_rate) # the number of trials we expect to mantain (inclusive)
                trials : list[optuna.Trial] = self._reduce_trials_due_to_results(surviving_trials, completed_results, n_trials_to_mantain)

                self.lg.writeLine(f"Step {i + 1} of successive halving: Expected to mantain {n_trials_to_mantain} from the original {len(results)} trials, mantained:")
                self.lg.writeLine(f"{[trial.number for trial in trials]},  from: {[trial.number for (trial, _) in results]}")

                if len(trials) < 1:
                    break


  
  
    def _run_true_hyperband(self):

        eta = self.get_input_value("hyperband_eta")
        min_steps = self.get_input_value("hyperband_min_steps")
        max_steps = self.n_steps

        self.max_steps_per_trial = self.get_input_value("max_steps_per_trial")

        s_max = int(math.log(max_steps / min_steps, eta))

        n_trials = self.get_input_value("initial_number_of_trials")

        if n_trials is not None:
            self.lg.writeLine(f"Initial number of trials passed of: {n_trials}")

        n_trials_done = 0

        for s in reversed(range(s_max + 1)):

            if n_trials is None:
                n_trials = int(math.ceil((s_max + 1) / (s + 1) * eta ** s))

            r = int(max_steps * eta ** (-s))

            trials = [self.study.ask() for _ in range(n_trials)]

            self._do_successive_halving(trials, r, s, eta)

            n_trials_done += n_trials

            n_trials = None

        return n_trials_done
    

  
    def _call_objective(self):

        n_trials_done = self._run_true_hyperband()

        if n_trials_done < self.n_trials:

            self.lg.writeLine(f"Hyperband did not complete asked number of trials ({self.n_trials})")

            self.n_trials = self.n_trials - n_trials_done

            self.lg.writeLine(f"Will run missing {self.n_trials} trials")

            super()._call_objective()


        


