import os
from concurrent.futures import ProcessPoolExecutor

# 数据集所在的目录
data_dirs = [
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\folder1',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\folder2',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\folder3',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\folder4',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\folder5'
]

output_dirs = [
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\tokenized_datasets_folder1',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\tokenized_datasets_folder2',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\tokenized_datasets_folder3',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\tokenized_datasets_folder4',
    'C:\\Users\\o3742\\Desktop\\literature-books-master\\tokenized_datasets_folder5'
]

script_path = 'C:\\Users\\o3742\\Desktop\\literature-books-master\\python\\process_folder.py'

def process_folder(data_dir, output_dir):
    os.system(f'python {script_path} {data_dir} {output_dir}')

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_folder, data_dir, output_dir) for data_dir, output_dir in zip(data_dirs, output_dirs)]
        
        for future in futures:
            future.result()  # 等待所有任務完成

    print("All folders processed and saved.")
