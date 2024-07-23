import os
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor
from token_processor import TokenFrequenciesProcessor, CustomChineseTokenizer, read_file_with_multiple_encodings, load_and_split_text_files, split_dataset, tokenize_batch, load_vocab_from_txt



# Function to process each folder
def process_folder(data_dir, output_dir):
    os.system(f'python {"process_folder.py"} {data_dir} {output_dir}')

# Function to load the configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # Load configuration from the specified file
    config = load_config('config.yml')
    data_dirs = config['data_dirs']
    tokenized_datasets_dirs = config['tokenized_datasets_dirs']
    
    # Use ProcessPoolExecutor to run the process_folder function in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_folder, data_dir, output_dir) for data_dir, output_dir in zip(data_dirs, tokenized_datasets_dirs)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    print("All folders processed and saved.")
    
    # Use token_processor to make customized token table
    processor = TokenFrequenciesProcessor(tokenized_datasets_dirs)
    processor.process_folders()
    processor.clean_combined_counter()
    
    # Use CustomChineseTokenizer to make customized token table  
    texts = load_and_split_text_files(data_dirs)
    dataset = split_dataset(texts)
    vocab = load_vocab_from_txt()
    tokenizer = CustomChineseTokenizer(vocab)
    tokenized_datasets = dataset.map(lambda examples: tokenize_batch(examples, tokenizer), batched=True, remove_columns=["text"])
    tokenized_datasets.save_to_disk(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/data_preprocess"))