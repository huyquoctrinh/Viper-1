import yaml

def read_yaml_as_dict(yml_path):
    """
    Reads a YAML file and returns its contents as a dictionary.
    Args:
        file_path (str): The path to the YAML file.
    """

    with open(yml_path, 'r') as file:
        data_dict = yaml.safe_load(file)
    return data_dict
