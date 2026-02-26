from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import threading
import time
from automl.basic_components.component_group import RunnableComponentGroup, setup_component_group

from automl.component import InputSignature
from automl.hp_opt.hp_opt_strategies.workers.hp_worker import HyperparameterOptimizationWorkerIndexed
from automl.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import math

from automl.rl.evaluators.rl_std_avg_evaluator import ResultLogger
import optuna

from automl.basic_components.state_management import StatefulComponentLoader

from automl.loggers.logger_component import use_logger

from automl.basic_components.exec_component import StopExperiment
import pandas

MINIMUM_SLEEP = 1
SLEEP_INCR_RATE = 2
MAX_SLEEP = 60

#OPTUNA_STUDY_PATH = "journal.log"


class HyperparameterOptimizationPipelineLoaderDetached(HyperparameterOptimizationPipeline):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         "trainings_at_a_time" : InputSignature(default_value=6),
                         "trainings_per_configuration" : InputSignature(default_value=3),
                         "stop_gracefully_wait_time" : InputSignature(default_value=3600),
                         "use_best_component_strategy" : InputSignature(default_value=True)
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

        self.use_best_component_strategy = self.get_input_value("use_best_component_strategy")


        if self.use_best_component_strategy:
            self.lg.writeLine(f"The strategy on using the multiple components will be to take the best of the first step and only continue with that")

        else:
            self.lg.writeLine(f"The strategy on using multiple components will be to train in parallel all the steps")


        # semaphore to use the evaluator
        self.evaluator_sem = threading.Semaphore(1)

        # a list of [trial_number, sem, configurations_not_attributed_to_worker], to use in prioritizing runn
        self.trainings_remaining_per_thread_sem = threading.Semaphore(1)
        self.trainings_remaining_per_thread = [None] * self.trainings_at_a_time

        self.optuna_usage_sem = threading.Semaphore(1)    

        self._setup_workers()

        self.stop_gracefully_wait_time = self.get_input_value("stop_gracefully_wait_time")

        self.__experiment_can_gracefuly_stop = False

        self.lg.writeLine(f"Finished processing input related to multiple component, detached execution")




    # THREADS SETUP ---------------------------------------------------

    def _setup_workers(self):

        self.workers : list[HyperparameterOptimizationWorkerIndexed] = []

        self.lg.writeLine(f"Total workers: {self.trainings_at_a_time}")

        for thread_index in range(self.trainings_at_a_time):

            worker = HyperparameterOptimizationWorkerIndexed(self, thread_index, report_max=self.use_best_component_strategy)

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

    def _generate_name_for_component_group(self, trial : optuna.Trial):
        
        return f"conf_{trial.number}"


    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
        
        with self.optuna_usage_sem:

            component_to_opt = super()._create_component_to_optimize(trial)

            group_directory = component_to_opt.get_artifact_directory()
            base_name = self._generate_name_for_component_group(trial)

            self.trial_loader_groups[trial.number] = setup_component_group(self.trainings_per_configuration, group_directory, base_name, component_to_opt, True)

            trial.set_user_attr("configuration_path", self.trial_loader_groups[trial.number].get_artifact_directory())

            return self.trial_loader_groups[trial.number]
        

    def _try_load_component_into_trial_loader_groups(self, trial : optuna.Trial):

        path_of_configuration = trial.user_attrs["configuration_path"]
        
        if os.path.exists(path_of_configuration):

            self.trial_loader_groups[trial.number] = RunnableComponentGroup({
                "artifact_relative_directory" : '',
                "base_directory" : path_of_configuration,
                "create_new_directory" : False
            })

            loaders = []

            for i in range(self.trainings_per_configuration):
                loaders.append(
                    StatefulComponentLoader(
                        {"base_directory" : self.trial_loader_groups[trial.number].get_artifact_directory(),
                         "create_new_directory" : False,
                         "artifact_relative_directory" : f'{i}'
                         }
                    )
                )
                
                self.trial_loader_groups[trial.number].define_component_as_child(loaders[i])
                

            self.trial_loader_groups[trial.number].pass_input({"components_loaders_in_group" : loaders}) 

            return self.trial_loader_groups[trial.number]
        
        else:
            return None
    

    def _get_loader_component_group(self, trial : optuna.Trial) -> RunnableComponentGroup:
                
        component_group : RunnableComponentGroup = self.trial_loader_groups[trial.number]
        
        return component_group


    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
         
        if not trial.number in self.trial_loader_groups.keys():
            
            return self._create_component_to_optimize(trial)
        
        else: 
            return self._get_loader_component_group(trial)

    
    # RUNNING THE WORKERS ------------------------------------------------------

    def _get_component_index_of_trial(self, trial : optuna.Trial, component_index=None, worker : HyperparameterOptimizationWorkerIndexed =None):

        if component_index is not None:
            return component_index
        
        else:
            to_return = trial.user_attrs.get("component_index_to_continue_using")

            if to_return is None:
                raise Exception(f"Trial {trial.number} does not have a component index specified and component index passed is None")
            
            if worker is not None:
                worker.thread_logger.writeLine(f"Component index to use for trial {trial.number} is {to_return}")

            return to_return

        

    def _run_available_worker(self, trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

        '''Waits until it can aquire a worker and then runs a job with it, freeing it in the end'''

        worker = self._aquire_available_worker()

        to_return = self._run_worker(worker, trial, component_index, n_steps, should_end, index_in_trainings_remaining)

        return to_return
    
    
    def _run_available_worker_till_step(self, trial : optuna.Trial, component_index, last_step, should_end, index_in_trainings_remaining):

        '''Waits until it can aquire a worker and then runs a job with it, freeing it in the end'''

        #self.lg.writeLine(f"HERE IN _run_available_worker_till_step: {trial.number}")

        worker = self._aquire_available_worker()

        to_return = self._run_worker_till_step(worker, trial, component_index, last_step, should_end, index_in_trainings_remaining)

        return to_return
    
    def _verify_and_aquire_worker_not_running_on_full_configuration(self, worker, trial, list_where_worker_working):

        [_, sem, current_number] = list_where_worker_working

        with sem:
            
            if current_number <= 0:
                worker.free_worker()
                raise Exception(f"Worker {worker.thread_index} attributed to trial {trial.number}, when that trial had already all its configurations trained / being trained (missing {current_number} trainings)")
            
            list_where_worker_working[2] = current_number - 1

    def _let_go_worker_list(self, worker, trial, list_where_worker_working):

        [_, sem, current_number] = list_where_worker_working

        with sem:
           list_where_worker_working[2] = current_number + 1


    def _run_worker(self, worker : HyperparameterOptimizationWorkerIndexed , trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

            '''Runs a job with a worker and then frees it'''

            self._verify_and_aquire_worker_not_running_on_full_configuration(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])

            component_index = self._get_component_index_of_trial(trial, component_index, worker)

            try:
                to_return = worker.try_run_component_in_group_all_steps(trial, component_index, n_steps, 
                                                    self._get_loader_component_group(trial).get_loader(component_index), 
                                                    should_end)

            except Exception as e:
                should_end[0] = True # if we receive an exception, we flag that the training for this trial should end
                self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
                worker.free_worker()
                raise e

            self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
            worker.free_worker()
            return to_return
    

    
    def _run_worker_till_step(self, worker : HyperparameterOptimizationWorkerIndexed , trial : optuna.Trial, component_index, last_step, should_end, index_in_trainings_remaining):

            '''Runs a job with a worker and then frees it'''

            #self.lg.writeLine(f"HERE IN _run_worker_till_step: {trial.number}")

            # we remove a configuration
            [_, sem, _] = self.trainings_remaining_per_thread[index_in_trainings_remaining]
            with sem:
                current_number = self.trainings_remaining_per_thread[index_in_trainings_remaining][2]
                
                if current_number <= 0:
                    worker.free_worker()
                    raise Exception(f"Worker {worker.thread_index} attributed to trial {trial.number}, when that trial had already all its configurations trained / being trained")

                self.trainings_remaining_per_thread[index_in_trainings_remaining][2] = current_number - 1

            component_index = self._get_component_index_of_trial(trial, component_index, worker)

            try:
                to_return = worker.try_run_component_in_group_until_step(trial, component_index, last_step, self._get_loader_component_group(trial).get_loader(component_index), should_end)

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


    def _wait_for_all_workers_free(self, max_time=math.inf):

        '''Sleeps until all workers are free'''

        next_worker_index = 0

        sleep_time = MINIMUM_SLEEP

        time_speeped = 0

        while True:

            next_worker = self.workers[next_worker_index]

            if next_worker.is_busy(): # if they are busy, we reset our search and wait
                next_worker_index = 0
                time.sleep(sleep_time)
                time_speeped += sleep_time

                if time_speeped > max_time:
                    return False # we stop waiting

                sleep_time = min(sleep_time * SLEEP_INCR_RATE, MAX_SLEEP) # we increase the sleep time

            next_worker_index = next_worker_index + 1

            if next_worker_index >= len(self.workers): # we did a full search, and they were all not busy
                break

        return True


    def _wait_for_all_workers_free_so_experiment_can_gracefully_stop(self):

        '''Sleeps until all workers are free'''

        if self.__experiment_can_gracefuly_stop:
            return
        
        else:
            if self._wait_for_all_workers_free(self.stop_gracefully_wait_time):
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


    # CONCURRENT TRIAL LOGIC ---------------------------------------------------         

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

        if self.use_best_component_strategy:
            
            # Expecting one result per component for the last step
            if "component_index" not in last_step_for_this_trial.columns:
                raise Exception(
                    "Best component strategy requires 'component_index' column in results."
                )

            # Find row with maximum result
            idx_max = last_step_for_this_trial["result"].idxmax()
            best_row = last_step_for_this_trial.loc[idx_max]

            component_index_with_maximum_result = int(best_row["component_index"])
            best_result = best_row["result"]

            # Store component index so we can continue using it
            trial.set_user_attr("component_index_to_continue_using", component_index_with_maximum_result)

            return best_result

        else:

            n_results_for_step = len(last_step_for_this_trial)

            last_results = last_step_for_this_trial["result"].tolist()

            if n_results_for_step != self.trainings_per_configuration:
                raise Exception(f"Mismatch between number of trainings that should have completed for trial {trial.number} ({self.trainings_per_configuration}) and actual number: {n_results_for_step}")

            return sum(last_results) / len(last_results) 
    

    def _generate_futures_for_single_trial(self, trial: optuna.Trial, index_in_trainings_remaining, first_worker : HyperparameterOptimizationWorkerIndexed, 
                                           run_first_worker_fun, run_other_workers_fun, should_end):
    
        '''Generated a list of lists of futures for a single trial. Between lists of futures, waiting is expected'''

        list_of_futures = []

        component_index_to_use = trial.user_attrs.get("component_index_to_continue_using")

        if component_index_to_use is not None:
            
            futures = []

            first_worker.thread_logger.writeLine(f"Trial {trial.number} has registered component index {component_index_to_use} as the only one to be used")

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker,
                    trial,
                    component_index_to_use,
                    self.n_steps,
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            list_of_futures.append(futures)

        elif self.use_best_component_strategy: # we are using the best component strategy, with the component being ran for the first time

            first_worker.thread_logger.writeLine(f"Trial {trial.number} does not have registered component index as the only one to be used")

            futures = []

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker,
                    trial,
                    0,
                    1,
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            # remaining jobs go through dynamic worker acquisition
            for component_index in range(1, self.trainings_per_configuration):
                futures.append(
                    [
                        run_other_workers_fun,
                        trial,
                        component_index,
                        1,
                        should_end,
                        index_in_trainings_remaining
                    ]
                )

            list_of_futures.append(futures)

            if self.n_steps > 1:
                
                first_worker.thread_logger.writeLine(f"Noting that steps to run is higher than one ({self.n_steps}) but trial {trial.number} is being run by the first time")
                first_worker.thread_logger.writeLine(f"Will run first step for all components ({self.trainings_per_configuration}) and from there run only the best component")
                
                list_of_futures.append(
                    [
                        [
                            self._compute_result_for_trial_run, # this is so the best step is registered
                            trial
                        ]
                    ]
                )

                list_of_futures.append(
                    [
                        [
                            run_other_workers_fun,
                            trial,
                            None,
                            self.n_steps - 1,
                            should_end,
                            index_in_trainings_remaining
                        ]
                    ]
                )


        else:

            futures = []

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker,
                    trial,
                    0,
                    self.n_steps,
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            # remaining jobs go through dynamic worker acquisition
            for component_index in range(1, self.trainings_per_configuration):
                futures.append(
                    [
                        run_other_workers_fun,
                        trial,
                        component_index,
                        self.n_steps,
                        should_end,
                        index_in_trainings_remaining
                    ]
                )

            list_of_futures.append(futures)

        return list_of_futures
    
    
    def _execute_futures_of_single_trial(self, trial : optuna.Trial, futures, executor : ThreadPoolExecutor, should_end):

            exceptions_to_raise = []

            for i in range(len(futures)):
                if isinstance(futures[i], list):
                    futures[i] = executor.submit(*futures[i])

            # wait for all jobs
            for future in as_completed(futures):

                exc = future.exception()

                if exc is not None:

                    should_end[0] = True

                    executor.shutdown(wait=True, cancel_futures=True)

                    if isinstance(exc, StopExperiment):
                        # we should cancel futures, but let the currently working threads stop gracefully their proccesses
                        executor.shutdown(wait=True, cancel_futures=True)

                    else: # as it is either a pruned experiment or a fail, there is not need to continue other experiments
                        executor.shutdown(wait=False, cancel_futures=True) 

                    exceptions_to_raise.append(exc)

            executor.shutdown(wait=True)

            self._raise_exception_in_execution(exceptions_to_raise)



    def _run_concurrent_experiments_for_single_trial(self, trial: optuna.Trial, index_in_trainings_remaining, run_till_step=False):
    
        # we only advance when we manage to get a non busy worker
        first_worker = self._aquire_available_worker()

        self.__experiment_can_gracefuly_stop = False

        if run_till_step:
            run_first_worker_fun = self._run_worker_till_step
            run_other_workers_fun = self._run_available_worker_till_step

        else:
            run_first_worker_fun = self._run_worker
            run_other_workers_fun = self._run_available_worker

        with use_logger(first_worker.thread_logger):
            
            # this is to force create the component for the trial
            loader : RunnableComponentGroup = self.get_component_to_test(trial)
            loader.unload_all_components_with_retries(lg=first_worker.thread_logger)
            del loader

        should_end = [False]
        
        list_of_futures = self._generate_futures_for_single_trial(trial, index_in_trainings_remaining, first_worker,
                                                          run_first_worker_fun, run_other_workers_fun, should_end)

        for futures in list_of_futures:
            executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)
            self._execute_futures_of_single_trial(trial, futures, executor, should_end)


    def _acquire_index_in_trainings_remaining(self, trial: optuna.Trial):

        index_in_trainings_remaining = self._get_next_available_index_trainings() # we get an index to represent this trial (in theory, there should always be one available)
        
        with self.trainings_remaining_per_thread_sem:
            self.trainings_remaining_per_thread[index_in_trainings_remaining] = [trial.number, threading.Semaphore(1), self.trainings_per_configuration]

        return index_in_trainings_remaining
    
    
    def _let_go_of_index_in_trainings_remaining(self, trial : optuna.Trial, index_in_trainings_remaining):
    
            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None # we let go of the index



    def _run_optimization(self, trial: optuna.Trial, trial_run_function=None, trial_run_funs_params={}):

        '''Runs the optimization function for a trial (essentially the objective)'''

        if self.check_if_should_stop_execution_earlier(raise_exception=False):
            self.check_if_should_stop_execution_earlier()
    
        if trial_run_function is None:
            trial_run_function = self._run_concurrent_experiments_for_single_trial

        if trial.number > self.beneficted_trainings and trial.number < self.trainings_at_a_time:
            time.sleep(60) # the idea is to do the maximum number of works per trial before going to the next trial
    
        index_in_trainings_remaining = self._acquire_index_in_trainings_remaining(trial)

        try:
            trial_run_function(trial, index_in_trainings_remaining, **trial_run_funs_params)
            self._let_go_of_index_in_trainings_remaining(trial, index_in_trainings_remaining)

        except Exception as e:

            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None

            raise e

        return self._compute_result_for_trial_run(trial)
    
    
    def _run_single_trial(self, trial=None):

        '''Runs optimization for a single trial, returning itself, the value, and an exception if any'''

        if trial is None:
            with self.optuna_usage_sem:
                trial = self.study.ask()

        try:
            value = self.objective(trial)
            return trial, value, None

        except Exception as e:
            return trial, None, e
        
        
    def _try_resume_single_trial(self, trial : optuna.trial.FrozenTrial):

        '''Tries resuming a single trial'''

        try:
            value = self._run_optimization(trial, self._run_concurrent_experiments_for_single_trial, {"run_till_step" : True})
            return trial, value, None
        
        except Exception as e:
            return trial, None, e
        

    def _try_load_all_resumed_trials(self, trials : list[optuna.trial.FrozenTrial]):

        self.lg.writeLine(f"Trying to load loaders of {len(trials)} queued trials...")

        for trial in trials:
            if self._try_load_component_into_trial_loader_groups(trial) is not None:
                self.lg.writeLine(f"Loaded trial {trial.number} from disk")
            
            else:
                self.lg.writeLine(f"Could not find trial {trial.number} in disk")


    def _try_resuming_unfinished_trials(self, trials : list[optuna.trial.FrozenTrial]):

        super()._try_resuming_unfinished_trials(trials)

        self._try_load_all_resumed_trials(trials)

        self.run_trials(trials, running_method=self._try_resume_single_trial)


    def _raise_exception_in_execution(self, exceptions_to_raise, raise_exception=True):

        '''Done to check which exceptions were noted in execution of a single trial and raise experiments with priority'''

        if len(exceptions_to_raise) > 0:

            for exception_to_raise in exceptions_to_raise:
                if not isinstance(exception_to_raise, (StopExperiment, optuna.TrialPruned)):
                    if raise_exception:    
                        raise exception_to_raise
                    else:
                        return exception_to_raise

            for exception_to_raise in exceptions_to_raise:
                if not isinstance(exception_to_raise, StopExperiment):
                    if raise_exception:
                        raise exception_to_raise
                    else:
                        return exception_to_raise
            
            if raise_exception:
                raise exceptions_to_raise[0]
            else:
                return exception_to_raise        
        

    def _generate_futures_for_trials(self, executor : ThreadPoolExecutor, trials, running_method):
        futures = []

        if isinstance(trials, list):

            for trial in trials:
                futures.append(executor.submit(running_method, trial))

        elif isinstance(trials, int):
            for _ in range(trials):
                futures.append(executor.submit(running_method))

        else:
            raise Exception(f"Trials must be a list of trials or an integer specifying the number of trials")

        return futures
    
    def mark_trial_as_complete(self, trial, value):
                    self.lg.writeLine(f"Trial {trial.number} marked as completed with value: {value}")
                    self.values["trials_done_in_this_execution"] += 1
                    with self.optuna_usage_sem:
                        self.study.tell(trial, value)

    def run_trials(self, trials, running_method=None, steps_to_run=None, mark_trials_as_completed=True):

        '''
        
        @returns results of all trials, results of non pruned trials, non pruned trials
        '''
        
        if running_method is None:
            running_method = self._run_single_trial

        self.lg.writeLine(f"Running {len(trials) if isinstance(trials, list) else trials} trials with running method: {running_method.__name__}, steps to run: {self.n_steps if steps_to_run is None else steps_to_run}, mark trials as completed: {mark_trials_as_completed}")

        results = []
        executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)

        completed_results = []
        surviving_trials = []

        if steps_to_run is not None:
            old_n_steps = self.n_steps
            self.n_steps = steps_to_run

        futures = self._generate_futures_for_trials(executor, trials, running_method)

        exceptions_to_raise = [] # we delay the raising of an exception as to allow results to be properly dealt with

        for future in as_completed(futures):

            trial, value, exception = future.result()

            if exception is not None:

                if isinstance(exception, optuna.TrialPruned):
                    self.values["trials_done_in_this_execution"] += 1
                    with self.optuna_usage_sem:
                        self.study.tell(trial=trial, state=optuna.trial.TrialState.PRUNED)
                    self.lg.writeLine(f"Trial {trial.number} was pruned")

                elif isinstance(exception, StopExperiment):
                    executor.shutdown(wait=True, cancel_futures=True) # we wait for current trials to end but cancel those that have not started
                    exceptions_to_raise.append(exception)
                    self.lg.writeLine(f"Trial {trial.number} over due to signal to stop the experiment")
                
                else:
                    with self.optuna_usage_sem:
                        self.study.tell(trial=trial, state=optuna.trial.TrialState.FAIL)
                    
                    trial_component_path = trial.user_attrs.get("configuration_path")
                    if trial_component_path is not None:
                        self.on_general_exception_trial(exception, trial_component_path, trial)

                    if not self.continue_after_error:
                        self.lg.writeLine(f"There was an error in trial {trial.number}, as <continue_after_error> is set to false, will cancel other trials, error was: {exception}")
                        self.stop_execution_earlier()
                        executor.shutdown(wait=True, cancel_futures=True)
                        exceptions_to_raise.append(exception)

                    else:
                        self.lg.writeLine(f"Error in trial {trial.number}, as <continue_after_error> is set to true, will not cancel other trials, error was: {exception}")
                        self.values["trials_done_in_this_execution"] += 1 # we only count as a completed trial if it did not end the test
                    

            else:
                completed_results.append((trial, value))
                surviving_trials.append(trial)

                if mark_trials_as_completed:
                    self.mark_trial_as_complete(trial, value)


            results.append((trial, value))

        executor.shutdown(wait=True)

        if steps_to_run is not None:
            self.n_steps = old_n_steps

        self._raise_exception_in_execution(exceptions_to_raise)

        return results, completed_results, surviving_trials
        
    

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
        component_group.unload_all_components_with_retries()

    
    #def _initialize_database(self):
    #    
    #    self.database_path = os.path.join(self.get_artifact_directory(), OPTUNA_STUDY_PATH)  # Path to the SQLite database file
    #    
    #    self.lg.writeLine(f"Trying to initialize database in path: {self.database_path}")
#
    #    self.storage = JournalStorage(JournalFileBackend(file_path=self.database_path))

  

    def _call_objective(self):
        self.run_trials(self.n_trials)
        

        