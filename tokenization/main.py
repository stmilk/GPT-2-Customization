import os
from concurrent.futures import ProcessPoolExecutor

data_dirs = [
    'folder1',
    'folder2',
    'folder3',
    'folder4',
    'folder5'
]

output_dirs = [
    'tokenized_datasets_folder1',
    'tokenized_datasets_folder2',
    'tokenized_datasets_folder3',
    'tokenized_datasets_folder4',
    'tokenized_datasets_folder5'
]

script_path = 'process_folder.py'

def process_folder(data_dir, output_dir):
    os.system(f'python {script_path} {data_dir} {output_dir}')

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_folder, data_dir, output_dir) for data_dir, output_dir in zip(data_dirs, output_dirs)]
        
        for future in futures:
            future.result()  

    print("All folders processed and saved.")
