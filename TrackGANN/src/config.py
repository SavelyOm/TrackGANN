import yaml
import os
import sys
import argparse


def load_config(config_dir, config_name):
    ''' Loads the configuration file. '''

    config_path = os.path.join(config_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ImportConfig():
    ''' Reads the configuration for subsequent access to individual settings. 
    Returns both the configuration data and the configuration object. '''

    parser = argparse.ArgumentParser(description='Runing GANN')
    parser.add_argument('--config-dir', '-d', type=str, 
                        help='Path to folder with configs')
    parser.add_argument('--config_name', '-c', type=str, required=True,
                       help='Name of config, for example (my_config.yaml)')
    
    args = parser.parse_args()
    try:
        config = load_config(args.config_dir, args.config_name)
        return config, args
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error in config loading: {e}")
        sys.exit(1)

