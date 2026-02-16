import os
import threading
from automl.basic_components.exec_component import State
from automl.basic_components.state_management import StatefulComponentLoader
from automl.component import Component
from automl.core.exceptions import common_exception_handling
from automl.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline, Component_to_opt_type
from automl.loggers.logger_component import override_first_logger
from automl.loggers.result_logger import ResultLogger
import optuna
import pandas

class HyperparameterOptimizationWorkerIndexed():
    

    def __init__(self, parent_hp_pipeline : HyperparameterOptimizationPipeline, thread_index):
        
        self.parent_hp_pipeline : HyperparameterOptimizationPipeline = parent_hp_pipeline

        self.thread_logger = self.parent_hp_pipeline.lg.clone(input_for_clone={"name" : f"{thread_index}", "log_text_file" : f"thread_loggers\\thread_logger_{thread_index}.txt"})

        self.thread_index = thread_index

        self._is_busy = False
        self._is_busy_sem = threading.Semaphore(1) 

    def is_busy(self):

        with self._is_busy_sem:
            return self._is_busy

    def aquire_if_not_busy(self):

        with self._is_busy_sem:

            if not self._is_busy:
                self._is_busy = True
                return self
        
        return None
    
    def free_worker(self):

        with self._is_busy_sem:
            self._is_busy = False


    # LOADER RELATED ------------------------------------------------------------

    def _check_running_state_of_component(self, component_to_opt : Component):

        running_state = component_to_opt.values.get("running_state", None)

        if running_state is not None:

            if State.equals_value(running_state, State.ERROR):
                raise Exception(f"Error when component was got") 


    # OPTIMIZATION -------------------------------------------------------------------------
        
    
    def _load_component_to_test(self, component_loader : StatefulComponentLoader) -> Component_to_opt_type:
                
        component_to_opt = component_loader.get_component()

        self._check_running_state_of_component(component_to_opt)

        return component_to_opt
        

    
    def _try_run_single_component_in_group(self, trial : optuna.Trial, component_index, component_loader : StatefulComponentLoader):
        
        component_to_test_path = str(component_loader.get_artifact_directory())

        self._load_component_to_test(component_loader)

        try:
  
            component_loader.save_component()
            component_loader.unload_if_loaded()

            component_loader.detach_run_component(
                                                        to_wait=True, 
                                                        global_logger_level='INFO'
                                                        )
            
            self._load_component_to_test(component_loader)
            component_loader.unload_if_loaded()


        except Exception as e:
            self.on_exception_running_trial(e, component_to_test_path, trial, component_index)
            raise e
        

    def _get_last_reported_step(self, trial : optuna.Trial, component_index):

        with self.parent_hp_pipeline.results_logger_sem:
            
            results_logger: ResultLogger = self.parent_hp_pipeline.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            mask = (
                (df["experiment"] == trial.number) & 
                (df["component_index"] == component_index)
            )

            filtered_df = df.loc[mask]

            if len(filtered_df) == 0:
                return -1

            last_step = filtered_df["step"].max()

            return last_step
        
    
    def _load_and_report_resuts(self, trial : optuna.Trial, component_index : int, step : int, evaluation_results):

        with self.parent_hp_pipeline.results_logger_sem:
        
            # Log raw result
            self.parent_hp_pipeline.log_results_of_trial(trial, step, component_index, evaluation_results)

            # Count how many results we already have for this (trial, step)
            results_logger: ResultLogger = self.parent_hp_pipeline.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            mask = (
                (df["experiment"] == trial.number) &

                (df["step"] == step)
            )

            filtered_df = df.loc[mask]

            n_results_for_step = mask.sum()

            last_results = filtered_df["result"].tolist()

            enough_runs = n_results_for_step == self.parent_hp_pipeline.trainings_per_configuration

            if enough_runs:
                avg_result = sum(last_results) / len(last_results)    
                self.thread_logger.writeLine(f"Reporting result {avg_result}, average of results: {last_results}")
                trial.report(avg_result, step)

            else:
                self.thread_logger.writeLine(f"Not enough configurations to report, current is {n_results_for_step}, needed is {self.parent_hp_pipeline.trainings_per_configuration}. Current results are {last_results}")

        return enough_runs

    
    def try_run_component_in_group_in_not_busy_all_steps(self,  trial : optuna.Trial, component_index, n_steps, component_loader : StatefulComponentLoader, should_end):

        with self._is_busy_sem:

            if self._is_busy:
                return None
    
            self._is_busy = True

        to_return = self.try_run_component_in_group_all_steps(trial, component_index, n_steps, component_loader)

        with self._is_busy_sem:
            self._is_busy = False

        return to_return

    def try_run_component_in_group_in_not_busy(self,  trial : optuna.Trial, component_index, step, component_loader : StatefulComponentLoader):

        with self._is_busy_sem:

            if self._is_busy:
                return False
    
            self._is_busy = True

        self.try_run_component_in_group(trial, component_index, step, component_loader)

        with self._is_busy_sem:
            self._is_busy = False

        return True

    def try_run_component_in_group_all_steps(self, trial : optuna.Trial, component_index, n_steps, component_loader : StatefulComponentLoader, should_end):

        last_reported_step = self._get_last_reported_step(trial, component_index)

        if last_reported_step < 0 and self.parent_hp_pipeline.do_initial_evaluation:

            self.thread_logger.writeLine(f"Doing initial evaluation of trial {trial.number} before initiating training...")
            
            component_to_test_path = str(component_loader.get_artifact_directory())

            try:
                    component_to_test = self._load_component_to_test(component_loader)

            except Exception as e:
                    self.on_general_exception_trial(e, component_to_test_path, trial, component_index)


            with override_first_logger(self.thread_logger):
                    evaluation_results = self.parent_hp_pipeline._try_evaluate_component(component_to_test_path, trial, component_to_test)

            self._load_and_report_resuts(trial, component_index, 0, evaluation_results)

            self.thread_logger.writeLine(f"Results were: {evaluation_results}\n")

            del component_to_test
            component_loader.save_and_onload_component()
            

        if last_reported_step < 0:
            next_step = 1
        else:
            next_step = last_reported_step + 1

        self.thread_logger.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEPS TO DO {n_steps} -----------------------\n")
            

        for step in range(next_step, next_step + n_steps + 1):

            if should_end[0]:
                self.thread_logger.writeLine(f"Interruped in trial {trial.number}, in component {component_index}, before step {step}")
                break

            to_return = self.try_run_component_in_group(trial, component_index, step, component_loader)

        return to_return

    def try_run_component_in_group(self, trial : optuna.Trial, component_index, step, component_loader : StatefulComponentLoader):

        tid = threading.get_ident()
        self.thread_logger.writeLine(f"Starting worker on OS thread {tid}....\n")

        self.thread_logger.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEP {step} -----------------------\n")

        component_to_test_path = str(component_loader.get_artifact_directory())

        self.thread_logger.writeLine(f"Component is in {component_to_test_path}, loading and running it...\n")

        try:

                self._try_run_single_component_in_group(trial, component_index, component_loader)

                try:
                    component_to_test = self._load_component_to_test(component_loader)

                except Exception as e:

                    self.on_general_exception_trial(e, component_to_test_path, trial, component_index)

                self.thread_logger.writeLine("Finished running component, evaluating it...\n")

                with override_first_logger(self.thread_logger):
                    evaluation_results = self.parent_hp_pipeline._try_evaluate_component(component_to_test_path, trial, component_to_test)

                try:

                    self.thread_logger.writeLine(f"Finished evaluating component, reported results were {evaluation_results}\n")

                    enough_runs = self._load_and_report_resuts(trial, component_index, step, evaluation_results)

                    self.thread_logger.writeLine(f"Ended step {step} for component {component_index} in trial {trial.number}\n") 

                    if enough_runs and trial.should_prune(): # we verify this after reporting the result
                        self.thread_logger.writeLine("Prunning current experiment due to pruner...\n")
                        trial.set_user_attr("prune_reason", "pruner")
                        raise optuna.TrialPruned()

                    return evaluation_results
                
                except Exception as e:

                    if isinstance(e, optuna.TrialPruned):
                        raise e
                    
                    else:
                        self.on_general_exception_trial(e, component_to_test_path, trial, component_index)

        except Exception as e:

                if isinstance(e, optuna.TrialPruned):
                    raise e # don't consume exception, let it pass, so optuna can deal with it
                                
                # if we reached here, it means an exception other than the trial being pruned was got
                self.thread_logger.writeLine(f"Error in trial {trial.number} at step {step}, prunning it...")
                trial.set_user_attr("prune_reason", "error")


                if self.parent_hp_pipeline.continue_after_error: # if we are to continue after an error, we count the trial simply as pruned, and let optuna deal with it
                    self.thread_logger.writeLine(f"Exception will make the trial be ignored and continue, exception was: {e}\n")
                    raise optuna.TrialPruned("error")
                
                else: # if not, we propagate the exception
                    self.thread_logger.writeLine("As <continue_after_error> was set to False, we end the Hyperparameter Optimization process and propagate the error to the caller\n")
                    raise e
                
        except KeyboardInterrupt as e:
            
                self.thread_logger.writeLine(f"User interrupted experiment in trial {trial.number} at step {step}, prunning it...\n")
                trial.set_user_attr("prune_reason", "user_interrupt")

                raise e
    

    def on_exception_evaluating_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index):

        self.thread_logger.writeLine(f"ERROR: CAN'T EVALUATE TRIAL {trial.number} in component index {component_index}")

        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_specific_path = "on_evaluate_error_report.txt"
        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.thread_logger.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.thread_logger, exception, error_report_path)

        raise exception
    


    def on_exception_running_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index):

        self.thread_logger.writeLine(f"ERROR RUNNING TRIAL {trial.number} in component index {component_index}")

        error_report_specific_path = "on_run_error_report.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.thread_logger.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.thread_logger, exception, error_report_path)

        raise exception
        
    def on_general_exception_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index):

        self.thread_logger.writeLine(f"ERROR IN TRIAL {trial.number} in component index {component_index}")

        error_report_specific_path = "general_exception.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.thread_logger.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.thread_logger, exception, error_report_path)

        raise exception
