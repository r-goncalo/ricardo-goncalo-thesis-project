from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time
from automl.basic_components.component_group import RunnableComponentGroup, setup_component_group

from automl.component import InputSignature, Component
from automl.core.exceptions import common_exception_handling
from automl.hp_opt.hp_opt_strategies.workers.hp_worker import HyperparameterOptimizationWorkerIndexed
from automl.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import math

from automl.rl.evaluators.rl_std_avg_evaluator import ResultLogger
import optuna

from automl.basic_components.state_management import StatefulComponentLoader

from automl.basic_components.state_management import save_state
 
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from automl.loggers.logger_component import LoggerSchema, override_first_logger, use_logger

from automl.basic_components.exec_component import State, StopExperiment
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

        self.workers : list[HyperparameterOptimizationWorkerIndexed] = []

        self.lg.writeLine(f"Total workers: {self.trainings_at_a_time}")

        for thread_index in range(self.trainings_at_a_time):

            worker = HyperparameterOptimizationWorkerIndexed(self, thread_index)

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


    def log_results_of_trial(self, trial : optuna.Trial, step : int, component_index : int, evaluation_results):

        result = evaluation_results["result"]

        results_to_log = {'experiment' : trial.number, "component_index" : component_index, "step" : step, **self._suggested_values_by_trials[trial.number], "result" : result}

        for key, value in results_to_log.items():
            results_to_log[key] = [value]

        self.log_results(results_to_log)  

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

        '''Waits until it can aquire a worker and then runs a job with it, freeing it in the end'''

        worker = self._aquire_available_worker()

        to_return = self._run_worker(worker, trial, component_index, n_steps, should_end, index_in_trainings_remaining)

        return to_return


    def _run_worker(self, worker : HyperparameterOptimizationWorkerIndexed , trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

            '''Runs a job with a worker and then frees it'''

            # we remove a configuration
            [_, sem, _] = self.trainings_remaining_per_thread[index_in_trainings_remaining]
            with sem:
                current_number = self.trainings_remaining_per_thread[index_in_trainings_remaining][2]
                
                if current_number <= 0:
                    worker.free_worker()
                    raise Exception(f"Worker {worker.thread_index} attributed to trial {trial.number}, when that trial had already all its configurations trained / being trained")

                self.trainings_remaining_per_thread[index_in_trainings_remaining][2] = current_number - 1

            try:
                to_return = worker.try_run_component_in_group_all_steps(trial, component_index, n_steps, self._get_loader_component_group(trial).get_loader(component_index), should_end)

            except Exception as e:
                should_end[0] = True # if we receive an exception, we flag that the training for this trial should end
                worker.free_worker()
                raise e
            
            worker.free_worker()
            return to_return



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


    def _wait_for_all_workers_free(self):

        '''Sleeps until all workers are free'''

        next_worker_index = 0

        sleep_time = MINIMUM_SLEEP

        while True:

            next_worker = self.workers[next_worker_index]

            if next_worker.is_busy(): # if they are busy, we reset our search and wait
                next_worker_index = 0
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * SLEEP_INCR_RATE, MAX_SLEEP) # we increase the sleep time

            next_worker_index = next_worker_index + 1

            if next_worker_index >= len(self.workers): # we did a full search, and they were all not busy
                break


    def _wait_for_all_workers_free_so_experiment_can_gracefully_stop(self):

        '''Sleeps until all workers are free'''

        if self.__experiment_can_gracefuly_stop:
            return
        
        else:
            self._wait_for_all_workers_free()
            self.__experiment_can_gracefuly_stop = True

            
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



    def _compute_result_for_trial_run(self, trial : optuna.Trial):

        '''Computes the result for a trial that was already run, to be used inside the objective'''

        with self.results_logger_sem:
        
            # Count how many results we already have for this (trial, step)
            results_logger: ResultLogger = self.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            mask = (
                (df["experiment"] == trial.number) 
            
            )

            df_for_this_trial = df.loc[mask]

        max_step = df_for_this_trial["step"].max()

        last_step_for_this_trial = df_for_this_trial[df_for_this_trial["step"] == max_step]

        n_results_for_step = len(last_step_for_this_trial)

        last_results = last_step_for_this_trial["result"].tolist()

        if n_results_for_step != self.trainings_per_configuration:
            raise Exception(f"Mismatch between number of trainings that should have completed for trial {trial.number} ({self.trainings_per_configuration}) and actual number: {n_results_for_step}")
        
        return sum(last_results) / len(last_results) 
    

    def _run_concurrent_experiments_for_single_trial(self, trial: optuna.Trial, index_in_trainings_remaining):
    
        # we only advance when we manage to get a non busy worker
        first_worker = self._aquire_available_worker()

        self.__experiment_can_gracefuly_stop = False

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

                if isinstance(exc, StopExperiment):
                    # we should cancel futures, but let the currently working threads stop gracefully their proccesses
                    executor.shutdown(wait=True, cancel_futures=True)

                else:
                    executor.shutdown(wait=False, cancel_futures=True)

                with self.trainings_remaining_per_thread_sem:
                    self.trainings_remaining_per_thread[index_in_trainings_remaining] = None # we let go of the index
                    
                raise exc

        executor.shutdown(wait=True)

    
    def _run_optimization(self, trial: optuna.Trial):

        try:
            self.check_if_should_stop_execution_earlier()
        
        except StopExperiment as e:
            self._wait_for_all_workers_free_so_experiment_can_gracefully_stop()
            raise e


        if trial.number > self.beneficted_trainings and trial.number < self.trainings_at_a_time:
            time.sleep(60) # the idea is to do the maximum number of works per trial before going to the next trial
    
        index_in_trainings_remaining = self._get_next_available_index_trainings() # we get an index to represent this trial (in theory, there should always be one available)
        self.trainings_remaining_per_thread[index_in_trainings_remaining] = [trial.number, threading.Semaphore(1), self.trainings_per_configuration]

        try:
            self._run_concurrent_experiments_for_single_trial(trial, index_in_trainings_remaining)
            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None # we let go of the index

        except Exception as e:
            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None

            if isinstance(e, StopExperiment): # if the reason the training ended was an interrupt, we first let all workers deal with it
                self._wait_for_all_workers_free()

            raise e
 

        return self._compute_result_for_trial_run(trial)
    

    def _check_if_should_stop_execution_earlier(self):
        should_stop = super()._check_if_should_stop_execution_earlier()

        if should_stop:
            return True
        
        else:
            stop_path = os.path.join(self.get_artifact_directory(), "__stop")
            return os.path.exists(stop_path)
                
    
    def _on_earlier_interruption(self):

        super()._on_earlier_interruption()

        self.lg.writeLine(f"Hyperparamter optimization proccess was interrupted")
        
        stop_path = os.path.join(self.get_artifact_directory(), "__stop")

        if os.path.exists(stop_path):
            try:
                os.remove(stop_path)
                self.lg.writeLine(f"Stop sign was succecsfuly removed")

            except FileNotFoundError:
                self.lg.writeLine(f"Tried deleting stop signal, but was not there")
                
            except OSError as e:
                self.lg.writeLine(f"WARNING: failed to remove stop signal at {stop_path}: {e}")



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
        