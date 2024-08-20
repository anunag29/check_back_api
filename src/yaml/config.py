import yaml
import os

from src.logger.logger import Logger

log = Logger("account_ocr_service", "config", "../logs", 10)

class Config:
    
    def read_yaml(file_path):
        with open(file_path, 'r') as yaml_file:
            try:
                data = yaml.safe_load(yaml_file)
                return data
            except yaml.YAMLError as ex:
                log.error(f"Error reading YAML file", ex)
            return None

    def get(key):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        project_dir = os.path.dirname(src_dir)
        # Construct the relative path to application.yaml
        file_path = os.path.join(project_dir, 'application.yaml')
        yaml_data = Config.read_yaml(file_path)
        if yaml_data:
            keys = key.split(".")
            try:
                # Accessing values from the YAML data
                value = yaml_data
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return None
        else:
            log.warning("Failed to read YAML file.")