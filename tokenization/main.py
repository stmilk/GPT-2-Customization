import os
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor

# Function to process each folder
def process_folder(data_dir, output_dir, script_path):
    os.system(f'python {script_path} {data_dir} {output_dir}')

# Function to load the configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # Load configuration from the specified file
    config = load_config('config.yml')
    data_dirs = config['data_dirs']
    output_dirs = config['output_dirs']
    script_path = config['script_path']

    # Use ProcessPoolExecutor to run the process_folder function in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_folder, data_dir, output_dir, script_path) for data_dir, output_dir in zip(data_dirs, output_dirs)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print("All folders processed and saved.")
