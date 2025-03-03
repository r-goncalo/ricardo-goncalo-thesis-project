
import pandas as pd
import optuna


def load_study_from_dataframe(df : pd.DataFrame, study=None) -> optuna.Study:

    '''
    Created and returns an optuna trial given the values on the dataframe
    It can also be used to add trials to a given dataframe
    '''

    if study == None:
        study = optuna.create_study()

    # Iterate through the rows of the DataFrame to create trials
    for _, row in df.iterrows():
        # Create trial parameters from the DataFrame
        params = {col: row[col] for col in df.columns if col not in ['experiment', 'result']}  # Exclude non-parameter columns

        # Create the trial object
        trial = optuna.trial.create_trial(
            params=params,         # The trial parameters (hyperparameters)
            value=row['result'],    # The objective value (objective function result)
        )

        # Add the trial to the study
        study.add_trial(trial)
