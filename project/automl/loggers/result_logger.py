import os
import pandas as pd
from automl.component import InputSignature, Component, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


import wandb

import torch

import pandas
from typing import Dict

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np

RESULTS_FILENAME = 'results.csv'

class ResultLogger(LoggerSchema):

    # TODO: empty parameters_signature should be able to be removed
    # TODO: verify order at which keys are verified
    parameters_signature = {
            "results_filename" : InputSignature(default_value=RESULTS_FILENAME, description="The filename of the results file, in this case a csv file"),
            "results_columns" : InputSignature(possible_types=[list], description="The columns (metrics) of the results", mandatory=False),
            "save_results_on_log" : InputSignature(default_value=True, description="If the results should be save on the disk after each log")
        } 
    

    # INITIALIZATION --------------------------------------------------------

    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super().proccess_input_internal()
        
        self.results_filename = self.input["results_filename"]
        
        self._initialize_dataframe()
        
        self.save_on_log = self.input["save_results_on_log"]
    
    
    def _initialize_dataframe(self):
        
        try:
        
            dataframe_on_folder = self.loadDataframe(filename=self.results_filename)

            self.dataframe = dataframe_on_folder
            self.columns = self.dataframe.columns
            self._writeLine(f"Results dataframe with filename {self.results_filename} already existed with columns {self.columns}")
        
        except Exception as e:
                        
            self.columns = self.input["results_columns"]
            self.dataframe = pandas.DataFrame(columns=self.columns)
            self._save_dataframe()
            
        


                
        
    # USAGE -------------------------------------------------------------------
 
 
    @requires_input_proccess           
    def log_results(self, results : Dict[str, list]):
                
        for key, value in results.items(): # TODO: WE SHOULD JUST MAKE results = [results] and it would work
            if not isinstance(value, list):
                results[key] = [results[key]]
                
        results_df = pandas.DataFrame(results, columns=self.columns)
        
        self.dataframe = pandas.concat((self.dataframe, results_df), ignore_index=True) 
        
        if self.save_on_log:
            self._save_dataframe()
              
        
    
    @requires_input_proccess    
    def save_dataframe(self):
        self._save_dataframe()
                
    def _save_dataframe(self):
        
        '''For internal use, does not trigger input processing'''
        
        self.saveDataframe(self.dataframe, filename=self.results_filename)
        
    
    @requires_input_proccess
    def get_results_columns(self):
        return self.columns
        

   # RETURN RESULTS ------------------------------------------------------------------------------------------------
    
    @requires_input_proccess
    def get_number_of_rows(self):
        
        return len(self.dataframe)
        
    @requires_input_proccess
    def get_last_results(self):
                        
        return self.dataframe.tail(1).to_dict(orient="records")[0]
    
    @requires_input_proccess
    def get_n_last_results(self, n_results):
        
        return self.dataframe.tail(n_results).to_dict(orient="records")[0]
    
    @requires_input_proccess
    def get_avg_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).mean()
    
    
    @requires_input_proccess
    def get_std_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).std()
    
    @requires_input_proccess
    def get_avg_and_std_n_last_results(self, n_results, column):
        
        return self.get_avg_n_last_results(n_results, column), self.get_std_n_last_results(n_results, column)
    
    @requires_input_proccess
    def get_sorted_dataframe(self, column, ascending):
        
        return self.dataframe.sort_values(by=column, ascending=ascending)
    
    @requires_input_proccess
    def get_n_last_ordered_results(self, n, column, ascending = True):
        
        '''Returns the best n results'''
        
        ordered_dataframe = self.get_sorted_dataframe(column=column, ascending=ascending)
        
        return ordered_dataframe.tail(n).to_records()
    
    @requires_input_proccess
    def get_dataframe(self):
        return self.dataframe

    def get_grouped_dataframes(self, key_to_group_by) -> Dict[str, pandas.DataFrame]:
        
        grouped_dataframes = self.dataframe.groupby(key_to_group_by)

        grouped_dataframes_dict = {k: v for k, v in grouped_dataframes}
        
        return grouped_dataframes_dict

    # GRAPHS ------------------------------------------------------------------------------------------------------------------

    @requires_input_proccess
    def plot_bar_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None, to_show=True, y_label=''):
        """
        Plots a graph using the dataframe stored in ResultLogger.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param save_path: Optional path to save the plot as an image.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))


        if isinstance(y_axis, str):
            y_axis = [y_axis]
            
        for i in range(len(y_axis)):

            if y_axis[i] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i]}' not found in dataframe.")
            
        categories = [str(value) for value in self.dataframe[x_axis]]


        for column_name in y_axis:
            plt.bar(self.dataframe[x_axis], self.dataframe[column_name])

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()


    @requires_input_proccess
    def plot_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None, to_show=True, y_label=''):
        """
        Plots a graph using the dataframe stored in ResultLogger.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param save_path: Optional path to save the plot as an image.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))


        if isinstance(y_axis, str):
            y_axis = [y_axis]
            
        for i in range(len(y_axis)):
            
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0] }' not found in dataframe.")


        #plt.figure(figsize=(10, 6))
        for (column_name, name_to_plot) in y_axis:
            plt.plot(self.dataframe[x_axis], self.dataframe[column_name], marker='o', label=name_to_plot)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()
            
    @requires_input_proccess
    def plot_confidence_interval(self, x_axis : str, y_column : str, y_values_label : str = 'mean', show_std=True, aggregate_number = None, title : str = '', save_path: str = None, to_show=True, y_label='', x_slice_range=None):

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        if y_column not in self.dataframe.columns:
            raise KeyError(f"Column '{y_column}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        x_values = self.dataframe[x_axis]
        values = self.dataframe[y_column]

        if x_slice_range is not None:
            x_values = x_values[x_slice_range[0]:max(x_slice_range[1], len(x_values))]
            values = values[x_slice_range[0]:max(x_slice_range[1], len(x_values))]

        if aggregate_number is not None:

            aggregated_values = [ [ values[u - aggregate_number + i] for i in range(0, aggregate_number * 2) ] for u in range(aggregate_number, len(values) - aggregate_number) ]
            x_values = x_values[aggregate_number:(len(x_values) - aggregate_number)]

            mean_values = np.mean(aggregated_values, axis=1)
        
            if show_std:
                std_values = np.std(aggregated_values, axis=1)

        else: # aggregate number is None
            mean_values = values
            std_values = None
        

        plt.plot(x_values, mean_values, label=y_values_label)
        
        if show_std and std_values is not None:
            plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.3)


        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.get_artifact_directory() + '\\' + save_path)

        if to_show:
            plt.show()

    
    @requires_input_proccess
    def plot_linear_regression(self, x_axis: str, y_axis: list, title: str = '', save_path: str = None, to_show=True, y_label=''):
       """
       Plots a graph with linear regression lines for the given columns in the dataframe.

       :param x_axis: The column key for the X-axis.
       :param y_axis: A list of column keys for the Y-axis.
       :param title: Optional title for the plot.
       :param save_path: Optional path to save the plot as an image.
       :param to_show: Whether to display the plot.
       :param y_label: Label for the Y-axis.
       """
       if self.dataframe.empty:
           raise ValueError("Dataframe is empty. Log results before plotting.")

       if x_axis not in self.dataframe.columns:
           raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

       if isinstance(y_axis, str):
           y_axis = [y_axis]
               
       for i in range(len(y_axis)):
           if isinstance(y_axis[i], str):
               y_axis[i] = (y_axis[i], y_axis[i])

           if y_axis[i][0] not in self.dataframe.columns:
               raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")
       
       for (column_name, name_to_plot) in y_axis:
           # Extract X and Y data
           X = self.dataframe[[x_axis]].values  # X values (reshaped for regression)
           Y = self.dataframe[column_name].values  # Y values

           # Fit the linear regression model
           model = LinearRegression()
           model.fit(X, Y)

           # Predict Y values from the linear model
           Y_pred = model.predict(X)

           # Plot the regression line
           plt.plot(self.dataframe[x_axis], Y_pred, label=f'{name_to_plot} Regression Line')

       plt.xlabel(x_axis)
       plt.ylabel(y_label)

       if title != '':
           plt.title(title)

       plt.legend()
       plt.grid(True)

       if save_path:
           plt.savefig(self.logDir + '\\' + save_path)

       if to_show:
           plt.show()


    def plot_polynomial_regression(
        self,
        x_axis: str,
        y_axis: list,
        degrees: int | list = 2,
        title: str = '',
        save_path: str = None,
        to_show=True,
        y_label=''
    ):
        """
        Plots polynomial regression curves for the given columns in the dataframe.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis. Can be str or [(col_name, plot_label)].
        :param degrees: Polynomial degree or list of degrees (int or list[int]).
        :param title: Optional title for the plot.
        :param save_path: Optional path to save the plot as an image.
        :param to_show: Whether to display the plot.
        :param y_label: Label for the Y-axis.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: {list(self.dataframe.columns)}")

        if isinstance(y_axis, str):
            y_axis = [y_axis]

        for i in range(len(y_axis)):
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")

        if isinstance(degrees, int):
            degrees = [degrees]  # normalize to list

        X = self.dataframe[[x_axis]].values
        X_sorted = np.sort(X, axis=0)  # ensures smooth curves

        for (column_name, name_to_plot) in y_axis:
            Y = self.dataframe[column_name].values

            for deg in degrees:
                # Create polynomial regression pipeline
                model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
                model.fit(X, Y)

                # Predict on sorted X for smooth curve
                Y_pred = model.predict(X_sorted)

                # Plot regression curve
                plt.plot(X_sorted, Y_pred, label=f'{name_to_plot} (degree {deg})')


        plt.xlabel(x_axis)
        plt.ylabel(y_label)

        if title:
            plt.title(title)

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()

    @requires_input_proccess
    def plot_piecewise_linear_regression(self, x_axis: str, y_axis: list, n_segments: int,
                                     title: str = '', save_path: str = None, to_show=True, y_label='', plot_with_segment_names=False):
        """
        Plots a graph with piecewise linear regression lines for the given columns in the dataframe.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param n_segments: Number of segments to split the data into.
        :param title: Optional title for the plot.
        :param save_path: Optional path to save the plot as an image.
        :param to_show: Whether to display the plot.
        :param y_label: Label for the Y-axis.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: {self.dataframe.columns}")

        if isinstance(y_axis, str):
            y_axis = [y_axis]

        for i in range(len(y_axis)):
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")

        # Sort by x_axis to make clean splits
        df_sorted = self.dataframe.sort_values(by=x_axis).reset_index(drop=True)
        X_all = df_sorted[[x_axis]].values

        x_min, x_max = df_sorted[x_axis].min(), df_sorted[x_axis].max()
        segment_bounds = np.linspace(x_min, x_max, n_segments + 1)

        for (column_name, name_to_plot) in y_axis:
            Y_all = df_sorted[column_name].values

            ax = plt.gca()
            next_color = ax._get_lines.get_next_color()

            for i in range(n_segments):
                seg_mask = (df_sorted[x_axis] >= segment_bounds[i]) & (df_sorted[x_axis] < segment_bounds[i+1])
                X_seg = df_sorted.loc[seg_mask, [x_axis]].values
                Y_seg = df_sorted.loc[seg_mask, column_name].values

                if len(X_seg) < 2:
                    continue  # Skip segments without enough points

                # Fit linear regression for the segment
                model = LinearRegression()
                model.fit(X_seg, Y_seg)

                # Predict on the segment
                Y_pred = model.predict(X_seg)

                # Plot regression line
                if plot_with_segment_names:
                    plt.plot(X_seg, Y_pred, color=next_color, label=f'{name_to_plot} Seg {i+1}')
                else:   
                    plt.plot(X_seg, Y_pred, color=next_color)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)

        if title:
            plt.title(title)

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '/' + save_path)

        if to_show:
            plt.show()


    def plot_current_graph(self, title: str = None, y_label : str = None):
        if title is not None:
            plt.title(title)

        if y_label != None:
            plt.ylabel(ylabel=y_label)

        plt.show()


 
def get_results_logger_from_file(folder_path, results_filename=RESULTS_FILENAME) -> ResultLogger:
        
    artifact_base_directory = folder_path
    artifact_relative_directory = ''
    
    datafrane = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))
    
    results_columns = datafrane.columns.tolist()    
    
    resuls_logger = ResultLogger(
        {
            "artifact_relative_directory": artifact_relative_directory,
            "base_directory": artifact_base_directory,
            "create_new_directory": False,
            "results_filename": results_filename,
            "results_columns": results_columns
        }
    )
    
    return resuls_logger
        
            
def aggregate_results_logger(paths, new_directory, new_column=None, results_filename=RESULTS_FILENAME, new_results_filename=None) -> ResultLogger:
    
    '''
    new_column is a tuple (column_name, [values for each dataframe loaded])
    '''
    if new_results_filename == None:
        new_results_filename = results_filename


    datafrane = None
    
    for folder_path_index in range(len(paths)):
        
        folder_path = paths[folder_path_index]
        
        artifact_base_directory = folder_path
        artifact_relative_directory = ''

        if datafrane is None:
            datafrane = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))
            
            if new_column is not None:
                datafrane[new_column[0]] = new_column[1][folder_path_index]
                            
        else:
            
            loaded_dataframe = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))

            if new_column is not None:
                loaded_dataframe[new_column[0]] = new_column[1][folder_path_index]
                            
            datafrane = pd.concat([datafrane, loaded_dataframe])

    results_columns = datafrane.columns.tolist() 
    
    resuls_logger = ResultLogger(
        {
            "artifact_relative_directory": '',
            "base_directory": new_directory,
            "create_new_directory": False,
            "results_filename": new_results_filename,
            "results_columns": results_columns
        }
    )
    
    resuls_logger.saveDataframe(datafrane, filename=new_results_filename)
    
    return resuls_logger
    
        
    
    ## WANDB --------------------------------
    #
    #def initialize_wandb(self):
    #    
    #    self.writeLine("Initializing wandb...")
    #    
    #    self.wandb_run = wandb.init(project="rl_pipeline", entity="rl_pipeline", mode="offline", dir = self.logDir)
    #    
    #def log_to_wandb(self, toLog : dict):
    #    
    #    self.wandb_run.log(toLog)
    #    
    #def close_wandb(self):
    #
    #    self.writeLine("Closing wandb...")        
    #    self.wandb_run.finish()    
    
    
    