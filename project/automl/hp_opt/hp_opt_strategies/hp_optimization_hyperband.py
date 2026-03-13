

from automl.component import InputSignature
from automl.hp_opt.hp_opt_strategies.hp_optimization_loader_detached import HyperparameterOptimizationPipelineLoaderDetached

import math
import optuna



class HyperparameterOptimizationPipelineHyperband(HyperparameterOptimizationPipelineLoaderDetached):
    
    '''
    Executes hyperband algorithm brackets untill all trials are over
    '''

    parameters_signature = {
                         "hyperband_eta": InputSignature(default_value=2,
                                                         description="The ratio at which we divide the trials in the sucessive halving brackets"
                                                         ),

                         "hyperband_min_steps": InputSignature(default_value=0),

                         "initial_number_of_trials" : InputSignature(
                             mandatory=False,
                             description="Forces initial number of trials (of first sucessive halving bracket) to be this number"),
                                                }
    
    exposed_values = {"current_hyperband_bracket" : -1, "current_sucessive_halving_bracket" : -1} 
            
    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()

        self.eta = self.get_input_value("hyperband_eta")
        self.min_steps = self.get_input_value("hyperband_min_steps")

        self.max_steps = self.n_steps

        self.max_steps_to_do_in_hyperband = self.max_steps - self.min_steps

        self.n_initial_hyperband_trials = self.get_input_value("initial_number_of_trials")

        self.lg.writeLine(f"Finished processing input related to hyperband execution")


        
    # SUCESSIVE HALVING --------------------------------------------------------------------

    def _reduce_trials_due_to_results(self, results : list[tuple[optuna.Trial, int]], n_trials_to_mantain : int):

        '''
        Reduces the number of trials due to their results, knowing the number of trials we want to mantain
        '''

        if n_trials_to_mantain >= len(results): # in the case we don't have enough trials, we just return all
            return results

        results.sort(key=lambda x: x[1], reverse=(self.direction == "maximize")) # the best trials will be put at the begining (highest result earlier)

        trials_to_mantain : list[tuple[optuna.Trial, int]] = [(t, r) for t, r in results[:n_trials_to_mantain]]

        if len(trials_to_mantain) > 0: # if there are still trials to mantain, we mark as pruned the trials that were discarded

            for trial_to_discard, result in results[n_trials_to_mantain:]:
                self.mark_trial_as_pruned(trial_to_discard, result)

        else: # if no trials are mantained, then we mark all surviving trials as completed
            for trial, result in results:
                self.mark_trial_as_complete(trial, result)

        return trials_to_mantain
    

    def _resume_single_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate, i, total_step_budget):
            
            '''Resumes a sucessive halving bracket that was being done'''

            raise NotImplementedError() # this is not completed and not necessary, we can ignore this code

            n_trials_to_mantain = len(trials)

            total_step_budget = 0  
            end_successive_halving = False

            step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i)

            stepd_that_should_be_done = total_step_budget + step_budget

            self.lg.writeLine(
                    f"Resuming: [HB] Bracket {number_of_runs} | Rung {i} | "
                    f"{len(trials)} trials | steps={step_budget} | total_steps={stepd_that_should_be_done}\n"
                )
            
            self.lg.writeLine(f"Trials: {[trial.number for trial in trials]}")

            if step_budget + total_step_budget > self.max_steps:

                step_budget = self.max_steps - total_step_budget
                end_successive_halving = True

                self.lg.writeLine(f"Because of limit of max steps per trial of {self.max_steps} and current steps done {total_step_budget}, this will be last bracket with {step_budget} steps instead")

            results, completed_results = self.run_trials(trials, running_method=self._try_resume_single_trial, steps_to_run=total_step_budget + step_budget, mark_trials_as_completed=False)

            n_trials_to_mantain = max(0, n_trials_to_mantain // configurations_to_study_decrease_rate) # the number of trials we expect to mantain (inclusive)
            trials_and_results_to_mantain : list[tuple[optuna.Trial, int]] = self._reduce_trials_due_to_results(surviving_trials, completed_results, n_trials_to_mantain)

            trials = [t for t, _ in trials_and_results_to_mantain]
            
            self.lg.writeLine(f"Step {i + 1} of successive halving: Expected to mantain {n_trials_to_mantain} from the original {len(results)} trials, mantained:")
            self.lg.writeLine(f"{[trial.number for trial in trials]},  from: {[trial.number for (trial, _) in results]}\n")

            end_successive_halving = end_successive_halving or len(trials) < 1

            return trials_and_results_to_mantain, trials, end_successive_halving
    

    def _resume_successive_halving(self, trials, initial_resource_per_config, number_of_runs, configurations_to_study_decrease_rate):

        '''Resumes a sucessive halging algorithm resuming a bracket it was doing and then continuing it normally'''

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




    def _do_successive_halving(self, trials, 
                               initial_resource_per_config, 
                               number_of_runs, 
                               configurations_to_study_decrease_rate, 
                               initial_i=0,
                               initial_trials_to_mantain=None):
            
            # this is here in case the number of trials received by the function is less than the ones it is supposed to mantain, such as in the case of some of them having been pruned
            n_trials_to_mantain = len(trials) if initial_trials_to_mantain is None else initial_trials_to_mantain

            total_step_budget = 0  
            end_successive_halving = False

            for i in range(initial_i, number_of_runs + 1):

                self.values["current_sucessive_halving_bracket"] = i

                step_budget = int(initial_resource_per_config * configurations_to_study_decrease_rate ** i) # step budget is higher in later brackets, as we study less configurations
               
                self.lg.writeLine(
                    f"[HB] Bracket {number_of_runs} | Rung {i} | "
                    f"{len(trials)} trials | steps={step_budget}"
                )

                self.lg.writeLine(f"Trials: {[trial.number for trial in trials]}")

                if self.max_steps is not None and step_budget + total_step_budget > self.max_steps:

                    step_budget = self.max_steps - total_step_budget
                    end_successive_halving = True

                    self.lg.writeLine(f"Because of limit of max steps per trial of {self.max_steps} and current steps done {total_step_budget}, this will be last bracket with {step_budget} steps instead")

                # here we receive all results and the results of the trials that were not pruned 
                results, completed_results = self.run_trials(trials, steps_to_run=step_budget, mark_trials_as_completed=False)

                total_step_budget = total_step_budget + step_budget

                if end_successive_halving:
                    trials_and_results_to_mantain = completed_results # this is so this trials are later marked as completed
                    break

                else:
                    n_trials_to_mantain = max(0, n_trials_to_mantain // configurations_to_study_decrease_rate) # the number of trials we expect to mantain (inclusive)
                    trials_and_results_to_mantain : list[tuple[optuna.Trial, int]] = self._reduce_trials_due_to_results(completed_results, n_trials_to_mantain)
                
                trials = [t for t, _ in trials_and_results_to_mantain]

                self.lg.writeLine(f"Step {i + 1} of successive halving: Expected to mantain {n_trials_to_mantain} from the original {len(results)} trials, mantained:")
                self.lg.writeLine(f"{[trial.number for trial in trials]},  from: {[trial.number for (trial, _) in results]}\n")

                if len(trials) < 1 or end_successive_halving:
                    break

            for trial, result in trials_and_results_to_mantain: # if there are still trials at the end of the successive halving, we mark them as done
                self.mark_trial_as_complete(trial, result)

            self.values["current_sucessive_halving_bracket"] = -1

    # HYPERBAND ALGORITHM ----------------------------------------------------------


    def _resume_true_hyperband_bracket(self, s, trials):

        ''' Resumes an hyperband bracket'''

        r = int(self.max_steps * self.eta ** (-s))

        self._resume_successive_halving(trials, r, s, self.eta)


    def _resume_hyperband(self, trials):

        '''Resumes the hyperband algorithm with trials that were still being run'''

        self.lg.writeLine(f"Resuming hyperband algorithm, last bracket registered was: {self.values['current_hyperband_bracket']}")

        s_max = int(math.log(self.max_steps_to_do_in_hyperband, self.eta))

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
    
  
    def _run_true_hyperband_bracket(self, current_hyperband_bracket, n_trials):

        '''
        Does an hyperband bracket, running the minimum number of steps and then executing hyperband as normal
        '''

        r = int(self.max_steps_to_do_in_hyperband * self.eta ** (-current_hyperband_bracket))

        with self.optuna_usage_sem:
            trials = [self.study.ask() for _ in range(n_trials)]

        if self.min_steps > 0:

            self.lg.writeLine(f"Will first run each of the {n_trials} for {self.min_steps}")

            results, completed_results = self.run_trials(trials, mark_trials_as_completed=False, steps_to_run=self.min_steps)

            trials = [trial for trial, result in completed_results]

        self._do_successive_halving(trials=trials, 
                                    initial_resource_per_config=r, 
                                    number_of_runs=current_hyperband_bracket, 
                                    configurations_to_study_decrease_rate=self.eta,
                                    initial_trials_to_mantain=n_trials)


    def _run_true_hyperband(self, s_max=None):

        '''
        Runs an adapted version of hyperband
        First runs all sampled trials the minimum steps
        Only then does it start the hyperband configuration
        '''

        if s_max is None:
            s_max = int(math.log(self.max_steps_to_do_in_hyperband, self.eta))

        n_trials = self.n_initial_hyperband_trials

        if n_trials is not None:
            self.lg.writeLine(f"Initial number of trials passed of: {n_trials}")

        n_trials_done = 0

        for current_hyperband_bracket in reversed(range(s_max + 1)):

            self.values["current_hyperband_bracket"] = current_hyperband_bracket

            end_early = False

            if n_trials is None:
                n_trials = int(math.ceil((s_max + 1) / (current_hyperband_bracket + 1) * self.eta ** current_hyperband_bracket))

            trials_missing = self.n_trials - self.values["trials_done_in_this_execution"]

            if n_trials > trials_missing:
                self.lg.writeLine(f"Trials to do in bracket ({n_trials}) is higher than trials that are yet to be done: {trials_missing}, adapting it")
                n_trials = trials_missing
                end_early = True

            self._run_true_hyperband_bracket(current_hyperband_bracket, n_trials)

            n_trials_done += n_trials

            n_trials = None

            if end_early:
                break

        self.values["current_hyperband_bracket"] = -1

        return n_trials_done

    

  
    def _call_objective(self):

        self._run_true_hyperband()

        while self.n_trials >  self.values["trials_done_in_this_execution"]:

            self.lg.writeLine(f"Hyperband did not complete asked number of trials, only did {self.values['trials_done_in_this_execution']} of {self.n_trials}")

            self._run_true_hyperband()



        


