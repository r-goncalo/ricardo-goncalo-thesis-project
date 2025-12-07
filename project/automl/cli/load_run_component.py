




def generate_path(component_path, target_path_dir='', target_path_name=None):

    import shutil
    from automl.basic_components.artifact_management import open_or_create_folder

    target_path_dir = '' if target_path_dir is None else target_path_dir

    experiment_path = open_or_create_folder(target_path_dir, target_path_name, create_new=True) if target_path_name is not None else target_path_dir

    shutil.copytree(component_path, experiment_path, dirs_exist_ok=True)

    return experiment_path


def load_component(component_path):

    from automl.utils.json_utils.json_component_utils import gen_component_from_path

    component = gen_component_from_path(component_path)

    return component

def run_component(component):

    component.pass_input({"logger_input" : {"necessary_logger_level" : "INFO"}})
    component.run()


def main(component_path, target_dir=None, target_dir_name=None):

    if target_dir is not None or target_dir_name is not None:
        component_path = generate_path(component_path, target_dir, target_dir_name)
    
    component = load_component(component_path)
    run_component(component)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization pipeline.")

    
    parser.add_argument("--component_path", type=str, default='.', help="Path of experiment")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory where to put the component run")
    parser.add_argument("--target_dir_name", type=str, default=None, help="Last folder where to put the run")
    
    args = parser.parse_args()

    main(
        component_path=args.component_path,
        target_dir=args.target_dir,
        target_dir_name=args.target_dir_name
         )

    
