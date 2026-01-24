from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time
from automl.basic_components.component_group import RunnableComponentGroup, setup_component_group

from automl.component import InputSignature, Component
from automl.core.exceptions import common_exception_handling
from automl.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import math

from automl.rl.evaluators.rl_std_avg_evaluator import ResultLogger
import optuna

from automl.basic_components.state_management import StatefulComponentLoader

from automl.basic_components.state_management import save_state
 
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from automl.loggers.logger_component import LoggerSchema, override_first_logger, use_logger

from automl.basic_components.exec_component import State
import pandas

MINIMUM_SLEEP = 1
SLEEP_INCR_RATE = 2
MAX_SLEEP = 60


OPTUNA_STUDY_PATH = "journal.log"

class HyperparameterOptimizationPipelineLoaderDetached(HyperparameterOptimizationPipeline):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         "trainings_at_a_time" : InputSignature(default_value=6),
                         "trainings_per_configuration" : InputSignature(default_value=3)
                       }
            


    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
                
        self.trial_loader_groups : dict[str, RunnableComponentGroup] = {} # the groups for each trial, each group has the same hyperparameter and different seeds

        self.trainings_at_a_time = self.get_input_value("trainings_at_a_time") # training processes that can be done at a time

        self.trainings_per_configuration = self.get_input_value("trainings_per_configuration") # training processes per configuration, the result is the average

        # trainings wich we should prioritize running at the same time (they are the minimum trainings we'll be doing even if we try our best to minimize it)
        # we track this to distribute best our resources by this number of configurations, and give only sparse resource to other configurations
        # the priority should be to give resources to trials with the lowest index
        self.beneficted_trainings = math.ceil(self.trainings_at_a_time / self.trainings_per_configuration)

        self.lg.writeLine(f"For {self.trainings_per_configuration} trainings per configuration, with {self.trainings_at_a_time} trainings being done at a time, the minimum number of hyperparameter configurations being tested at a time should be {self.beneficted_trainings}")

        # semaphore to use the evaluator
        self.evaluator_sem = threading.Semaphore(1)

        # a list of [trial_number, sem, configurations_not_attributed_to_worker], to use in prioritizing runn
        self.trainings_remaining_per_thread_sem = threading.Semaphore(1)
        self.trainings_remaining_per_thread = [None] * self.trainings_at_a_time

        self.trial_creation_sem = threading.Semaphore(1) # this is to be sure no problem regarding the sampling or any other parts of component creation appear
    
        self._setup_workers()


    # THREADS SETUP ---------------------------------------------------

    def _setup_workers(self):

        self.workers : list[HyperparameterOptimizationWorker] = []

        self.lg.writeLine(f"Total workers: {self.trainings_at_a_time}")

        for thread_index in range(self.trainings_at_a_time):

            worker = HyperparameterOptimizationWorker(self, thread_index)

            self.workers.append(worker)

            self.lg.writeLine(f"Created worker {worker.thread_index}, its logging will be made at {worker.thread_logger.get_artifact_directory()}")

        self.results_logger_sem = threading.Semaphore(1) 

        self.lg.writeLine()

    

    def _try_evaluate_component(self, component_to_test_path, trial : optuna.Trial, component_to_test = None) -> float:

        # this is only to guarantee only one results is evaluated at the same time
        with self.evaluator_sem:
            return super()._try_evaluate_component(component_to_test_path, trial, component_to_test)

    def _setup_results_logger(self, parameter_names):
        self.add_to_columns_of_results_logger(["experiment", "component_index", "step", *parameter_names, "result"])


    # OPTIMIZATION -------------------------------------------------------------------------


    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
        
        with self.trial_creation_sem:

            component_to_opt = super()._create_component_to_optimize(trial)

            group_directory = component_to_opt.get_artifact_directory()
            base_name = f"conf_{trial.number}"

            self.trial_loader_groups[trial.number] = setup_component_group(self.trainings_per_configuration, group_directory, base_name, component_to_opt, True)

            return self.trial_loader_groups[trial.number]
    

    def _get_loader_component_group(self, trial : optuna.Trial) -> RunnableComponentGroup:
                
        component_group : RunnableComponentGroup = self.trial_loader_groups[trial.number]
        
        return component_group


    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
         
        if not trial.number in self.trial_loader_groups.keys():
            
            return self._create_component_to_optimize(trial)
        
        else: 
            return self._get_loader_component_group(trial)

    
    def _run_available_worker(self, trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):


        worker = self._aquire_available_worker()

        self._run_worker(worker, trial, component_index, n_steps, should_end, index_in_trainings_remaining)


    def _run_worker(self, worker, trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

            worker : HyperparameterOptimizationWorker = worker

            # we remove a configura
            [_, sem, _] = self.trainings_remaining_per_thread[index_in_trainings_remaining]
            with sem:
                current_number = self.trainings_remaining_per_thread[index_in_trainings_remaining][2]
                
                if current_number <= 0:
                    worker.free_worker()
                    raise Exception(f"Worker {worker.thread_index} attributed to trial {trial.number}, when that trial had already all its configurations trained / being trained")

                self.trainings_remaining_per_thread[index_in_trainings_remaining][2] = current_number - 1

            try:
                worker.try_run_component_in_group_all_steps(trial, component_index, n_steps, self._get_loader_component_group(trial).get_loader(component_index), should_end)

            except Exception as e:
                should_end[0] = True # if we receive an exception, we flag that the training for this trial should end
                
                worker.free_worker()
                raise e
            
            worker.free_worker()


    def _aquire_available_worker(self):

        '''Aquires the next available worker, waiting if necessary'''

        next_worker_index = 0

        sleep_time = MINIMUM_SLEEP

        while True:

            next_worker = self.workers[next_worker_index]
            worker = next_worker.aquire_if_not_busy()

            if worker is not None:
                return worker

            next_worker_index = next_worker_index + 1

            if next_worker_index >= len(self.workers): # we did a full search, we wait before doing it again
                next_worker_index = 0
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * SLEEP_INCR_RATE, MAX_SLEEP) # we increase the sleep time

            
    def _get_next_available_index_trainings(self):
        
        '''Gets an index to represent an optuna job'''

        while True:

            with self.trainings_remaining_per_thread_sem:
                
                for i in range(len(self.trainings_remaining_per_thread)):

                    if self.trainings_remaining_per_thread[i] is None:
                    
                        self.trainings_remaining_per_thread[i] = 0 # we change its value so no other search will find this index
                        return i
                    
            time.sleep(MINIMUM_SLEEP)

    
    def _wait_for_previous_trials_to_be_occupied(self, trial : optuna.Trial):

        '''If a trial should wait before trying to acquire workers, waits'''

        if trial.number < self.beneficted_trainings:
            return
        
        to_wait = True

        while to_wait:

            trials_to_receive_resources_before_this_one = 0
            to_wait = False
            
            with self.trainings_remaining_per_thread_sem:

                for i, value in enumerate(self.trainings_remaining_per_thread):

                    if value is not None and value != 0:
                        [trial_number, sem, jobs_to_go] = value

                        if trial_number < trial.number: # if there is a trial previous of this one

                            with sem:
                                jobs_to_go = self.trainings_remaining_per_thread[i][2]

                                if jobs_to_go >= 1: # and it has a configuration that is not yet being trained

                                    trials_to_receive_resources_before_this_one += 1

                                    # if the number of trials that are ahead of this one on wanting to aquire a resource is higher than the number of beneficted trainings
                                    if trials_to_receive_resources_before_this_one >= self.beneficted_trainings:
                                        to_wait = True
                                        break

                
            if to_wait:
                time.sleep(MAX_SLEEP) # we wait

        # if we reached this, we can stop waiting
            


    
    def _run_optimization(self, trial: optuna.Trial):

        if trial.number > self.beneficted_trainings and trial.number < self.trainings_at_a_time:
            time.sleep(60) # the idea is to do the maximum number of works per trial before going to the next trial
    
        index_in_trainings_remaining = self._get_next_available_index_trainings() # we get an index to represent this trial (in theory, there should always be one available)
        self.trainings_remaining_per_thread[index_in_trainings_remaining] = [trial.number, threading.Semaphore(1), self.trainings_per_configuration]

        # we only advance when we manage to get a non busy worker
        first_worker = self._aquire_available_worker()

        with use_logger(first_worker.thread_logger):
            
            # this is to force create the component for the trial
            loader : RunnableComponentGroup = self.get_component_to_test(trial)
            loader.unload_all_components()
            del loader
        
        futures = []
        executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)

        should_end = [False]

        futures.append(
            executor.submit(
                self._run_worker,
                first_worker,
                trial,
                0,
                self.n_steps,
                should_end,
                index_in_trainings_remaining
            )
        )

        # remaining jobs go through dynamic worker acquisition
        for component_index in range(1, self.trainings_per_configuration):
            futures.append(
                executor.submit(
                    self._run_available_worker,
                    trial,
                    component_index,
                    self.n_steps,
                    should_end,
                    index_in_trainings_remaining
                )
            )

        # wait for all jobs
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                raise exc

        executor.shutdown(wait=True)

        with self.trainings_remaining_per_thread_sem:
            self.trainings_remaining_per_thread[index_in_trainings_remaining] = None # we let go of the index

    

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

  

    def _call_objective(self):

        self.study.optimize( lambda trial : self.objective(trial), 
                       n_trials=self.n_trials,
                       n_jobs=self.trainings_at_a_time,
                       callbacks=[self.after_trial])
        


class HyperparameterOptimizationWorker():
    

    def __init__(self, parent_hp_pipeline : HyperparameterOptimizationPipelineLoaderDetached, thread_index):
        
        self.parent_hp_pipeline : HyperparameterOptimizationPipelineLoaderDetached = parent_hp_pipeline

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
        
    
    def _load_and_report_resuts(self, trial : optuna.Trial, results_to_log, step):


        with self.parent_hp_pipeline.results_logger_sem:
        
            # Log raw result
            self.parent_hp_pipeline.log_results(results_to_log)

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
                return False
    
            self._is_busy = True

        self.try_run_component_in_group_all_steps(trial, component_index, n_steps, component_loader)

        with self._is_busy_sem:
            self._is_busy = False

        return True

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

        for step in range(n_steps):

            if should_end[0]:
                self.thread_logger.writeLine(f"Interruped in trial {trial.number}, in component {component_index}, before step {step}")
                break

            self.try_run_component_in_group(trial, component_index, step, component_loader)

    def try_run_component_in_group(self, trial : optuna.Trial, component_index, step, component_loader : StatefulComponentLoader):

        tid = threading.get_ident()
        self.thread_logger.writeLine(f"Starting worker on OS thread {tid}....\n")

        self.thread_logger.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEP {step + 1} -----------------------\n")

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

                    result = evaluation_results["result"]

                    suggested_values_for_trial = self.parent_hp_pipeline._suggested_values_by_trials[trial.number]

                    results_to_log = {'experiment' : trial.number, "component_index" : component_index, "step" : step, **suggested_values_for_trial, "result" : result}

                    for key, value in results_to_log.items():
                        results_to_log[key] = [value]

                    enough_runs = self._load_and_report_resuts(trial, results_to_log, step)

                    self.thread_logger.writeLine(f"Ended step {step + 1} for component {component_index} in trial {trial.number}\n") 

                    if enough_runs and trial.should_prune(): # we verify this after reporting the result
                        self.thread_logger.writeLine("Prunning current experiment due to pruner...\n")
                        trial.set_user_attr("prune_reason", "pruner")
                        raise optuna.TrialPruned()

                    return evaluation_results
                
                except Exception as e:

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
