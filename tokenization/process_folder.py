import thulac
import os
import chardet
from datasets import Dataset, DatasetDict
import sys
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

def fullwidth_to_halfwidth(text):
    result = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)

def read_and_tokenize(file_path, max_length=512):
    encodings = ['utf-8', 'gb18030', 'gb2312', 'gbk']
    text = None
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                break
        except (UnicodeDecodeError, TypeError):
            continue
    if text is None:
        raise RuntimeError(f"Could not decode file {file_path} with available encodings.")

    text = fullwidth_to_halfwidth(text)
    sentences = text.split('\n')
    
    thu = thulac.thulac(seg_only=True)
    tokenized_sentences = [' '.join([word for word, tag in thu.cut(sentence.strip())]) for sentence in sentences if sentence.strip()]
    return tokenized_sentences

def load_and_tokenize_datasets(data_dir, max_length=512):
    all_texts = []
    skipped_files = []
    file_token_sets = defaultdict(set)

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    tokenized_sentences = read_and_tokenize(file_path, max_length)
                    all_texts.extend(tokenized_sentences)
                    tokens_in_file = set()
                    for sentence in tokenized_sentences:
                        tokens_in_file.update(sentence.split())
                    for token in tokens_in_file:
                        file_token_sets[token].add(file_path)
                    print(f"Successfully processed file: {file_path}")
                except (RuntimeError, UnicodeDecodeError) as e:
                    skipped_files.append(file_path)
                    print(f"Could not decode file {file_path}: {e}")

    return all_texts, skipped_files, file_token_sets

def split_datasets(all_texts, train_size=0.8, val_size=0.1):
    train_texts, temp_texts = train_test_split(all_texts, train_size=train_size, random_state=42)
    val_texts, test_texts = train_test_split(temp_texts, train_size=val_size/(1-train_size), random_state=42)
    return train_texts, val_texts, test_texts

def save_datasets(train_texts, val_texts, test_texts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'validation.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_texts))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_texts))
    
    print(f"Datasets saved to {output_dir}")

def save_token_frequencies(train_texts, val_texts, test_texts, file_token_sets, output_file):
    token_counter = Counter()
    
    for dataset in [train_texts, val_texts, test_texts]:
        for sentence in dataset:
            tokens = sentence.split()
            token_counter.update(tokens)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, freq in token_counter.items():
            file_count = len(file_token_sets[token])
            f.write(f"{token}: {freq}, in {file_count} files\n")
    
    print(f"Token frequencies saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_folder.py <data_dir> <output_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Starting processing for {data_dir}")
    all_texts, skipped_files, file_token_sets = load_and_tokenize_datasets(data_dir)
    
    train_texts, val_texts, test_texts = split_datasets(all_texts)
    
    save_datasets(train_texts, val_texts, test_texts, output_dir)
    
    token_freq_file = os.path.join(output_dir, 'token_frequencies.txt')
    save_token_frequencies(train_texts, val_texts, test_texts, file_token_sets, token_freq_file)
    
    print(f"Train texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    print(f"Test texts: {len(test_texts)}")
    print(f"Skipped files: {skipped_files}")
