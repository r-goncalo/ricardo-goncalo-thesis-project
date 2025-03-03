
import optuna


def load_study_from_database(database_path: str, study_name='experiment') -> optuna.Study:
    """
    Loads an Optuna study from the given SQLite database file.

    Args:
        database_path (str): The path to the SQLite database file.

    Returns:
        optuna.Study: The loaded Optuna study.
    """
    # Define the storage URI for SQLite database
    storage_uri = f"sqlite:///{database_path}"
    
    # Load the study from the storage URI
    study = optuna.load_study(study_name=study_name, storage=storage_uri)
    
    return study