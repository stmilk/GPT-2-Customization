import os
import json
from collections import Counter
import random
from transformers import GPT2Tokenizer
from datasets import Dataset, DatasetDict

class TokenFrequenciesProcessor:
    """
    A class to process token frequencies from multiple files and perform various operations 
    on the token data, including combining frequencies, cleaning, and saving the results.
    """
    def __init__(self, tokenized_datasets_dirs, low_frequency_threshold=50, min_document_appearance=15):
        """
        Initialize the TokenProcessor with the base directory containing the tokenized datasets.
        
        Args:
            tokenized_datasets_dirs (str): The base directory path where tokenized dataset folders are located.

            low_frequency_threshold (int): The threshold below which tokens are considered low frequency and will be removed. Default is 50.
            
            min_document_appearance (int): The minimum number of documents a token must appear in to be retained. Tokens appearing in fewer than this number of documents will be removed. Default is 15.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenized_datasets_folders = [os.path.join(base_dir, tokenized_datasets_folder) for tokenized_datasets_folder in tokenized_datasets_dirs] 
        self.low_frequency_threshold = low_frequency_threshold
        self.min_document_appearance = min_document_appearance
        self.combined_counter = Counter()
        self.combined_token_file_counter = Counter()
        self.output_file = "data/combined_token_frequencies.txt"

    def read_token_frequencies(self, file_path):
        """
        Read token frequencies from a given file and update the counters.

        Args:
            file_path (str): The path to the file containing token frequencies.

        Returns:
            tuple: Two Counters, one for token frequencies and one for file frequencies.
        """
        token_counter = Counter()
        token_file_counter = Counter()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for ch in range(len(line) - 1, -1, -1):
                    if line[ch] == "f":
                        final1 = ch - 1
                    elif line[ch] == "n":
                        start1 = ch + 2
                    elif line[ch] == ")":
                        final = ch
                    elif line[ch] == "(":
                        start = ch + 1
                        break
                token, freq = line[:start - 3], line[start:final]
                token_counter[token] = int(freq)
                token_file_counter[token] = int(line[start1:final1])
        return token_counter, token_file_counter

    def process_folders(self):
        """
        Process each folder to read token frequencies and update the combined counters.
        """
        for folder in self.tokenized_datasets_folders:
            token_freq_file = os.path.join(folder, 'token_frequencies.txt')
            if os.path.exists(token_freq_file):
                token_counter, token_file_counter = self.read_token_frequencies(token_freq_file)
                self.combined_counter.update(token_counter)
                self.combined_token_file_counter.update(token_file_counter)
            else:
                print(f"Token frequency file not found: {token_freq_file}")

    def is_chinese_char(self, ch):
        """
        Check if a character is a Chinese character.

        Args:
            ch (str): A single character.

        Returns:
            bool: True if the character is a Chinese character, False otherwise.
        """
        ranges = [
            (0x4E00, 0x9FFF),   # Basic Chinese characters
            (0x3400, 0x4DBF),   # Extension A
            (0x20000, 0x2A6DF), # Extension B
            (0x2A700, 0x2B73F), # Extension C
            (0x2B740, 0x2B81F), # Extension D
            (0x2B820, 0x2CEA1), # Extension E
            (0x2CEB0, 0x2EBE0), # Extension F
            (0xF900, 0xFAFF),   # Compatibility characters
            (0x2F800, 0x2FA1F)  # Compatibility extensions
        ]
        return any(start <= ord(ch) <= end for start, end in ranges)

    def is_english_char(self, ch):
        """
        Check if a character is an English character.

        Args:
            ch (str): A single character.

        Returns:
            bool: True if the character is an English character, False otherwise.
        """
        return ('a' <= ch <= 'z') or ('A' <= ch <= 'Z')

    def clean_combined_counter(self):
        """
        Clean the combined counter by removing tokens based on specific criteria:
        1. Tokens with a frequency less than or equal to self.low_frequency_threshold.
        2. Tokens longer than 1 character that contain non-Chinese characters.
        3. Tokens longer than 1 character that contain digits.
        4. Tokens that appear in fewer than self.min_document_appearance documents.
        5. Remove the tokens that meet the above criteria from the combined counter and add single characters not in the combined counter.
        """
        delet = []

        # 1. Remove tokens with frequency <= self.low_frequency_threshold
        for i in self.combined_counter:
            if self.combined_counter[i] <= self.low_frequency_threshold:
                delet.append(i)
        
        # 2. Remove tokens longer than 1 character containing non-Chinese characters
        for i in self.combined_counter:
            if len(i) > 1:
                for j in i:
                    if not self.is_chinese_char(j):
                        delet.append(i)
                        break

        # 3. Remove tokens longer than 1 character containing digits
        for i in self.combined_counter:
            if len(i) > 1:
                for j in i:
                    if j.isdigit():
                        delet.append(i)
                        break

        # 4. Remove tokens appearing in fewer than self.min_document_appearance documents
        for i in self.combined_token_file_counter:
            if self.combined_token_file_counter[i] < self.min_document_appearance:
                delet.append(i)

        # 5. Remove the tokens that meet the above criteria and add single characters not in the combined counter
        for i in delet:
            for j in i:
                if j not in self.combined_counter:
                    self.combined_counter[j] = 1
            del self.combined_counter[i]
        
        self.save_combined_counter(self.output_file)

    def save_combined_counter(self, output_file):
        """
        Save the combined token frequencies to a specified file.

        Args:
            output_file (str): The path to the output file.
        """
        output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            for token, freq in self.combined_counter.items():
                f.write(f"{token}: {freq}\n")

        print(f"Combined token frequencies saved to {output_file}")
    

class CustomChineseTokenizer:
    def __init__(self, vocab, special_tokens=None):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens or {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'unk_token': '[UNK]',
            'pad_token': '[PAD]'
        }
        self._add_special_tokens_to_vocab()
        self.unk_token_id = self.vocab[self.special_tokens['unk_token']]
        self.max_token_length = max(len(token) for token in self.vocab)  # 记录最长 token 的长度
    
    def _add_special_tokens_to_vocab(self):
        for token in self.special_tokens.values():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
    
    def tokenize(self, text):
        tokens = []
        start = 0
        while start < len(text):
            matched = False
            end = min(start + self.max_token_length, len(text))  # 计算当前可能的最长 token 的结束位置
            for i in range(end, start, -1):  # 从最长到最短搜索
                if text[start:i] in self.vocab:
                    tokens.append(text[start:i])
                    start = i
                    matched = True
                    break
            if not matched:
                tokens.append(self.special_tokens['unk_token'])
                start += 1
        return tokens
    
    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)
    
    def convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.special_tokens['unk_token'])
    
    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (vocab_file,)

    def __call__(self, text, padding=True, truncation=True, max_length=512):
        tokens = self.tokenize(text)
        token_ids = [self.convert_token_to_id(token) for token in tokens]

        if padding:
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids = token_ids + [self.vocab[self.special_tokens['pad_token']]] * padding_length
            elif padding_length < 0:
                token_ids = token_ids[:max_length]
        
        attention_mask = [1 if id != self.vocab[self.special_tokens['pad_token']] else 0 for id in token_ids]
        special_tokens_mask = [1 if token in self.special_tokens.values() else 0 for token in tokens]
        if padding_length > 0:
            special_tokens_mask = special_tokens_mask + [1] * padding_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'special_tokens_mask': special_tokens_mask
        }

def read_file_with_multiple_encodings(file_path):
    encodings = ['utf-8', 'gb18030', 'gb2312', 'gbk', 'UTF-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, TypeError):
            continue
    raise RuntimeError(f"Could not decode file {file_path} with available encodings.")

def load_and_split_text_files(data_dirs):
    texts = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for data_dir in data_dirs:
        data_dir = os.path.join(base_dir, data_dir)
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.txt'):
                    try:
                        file_path = os.path.join(root, filename)
                        file_content = read_file_with_multiple_encodings(file_path)
                        lines = file_content.split('\n')
                        lines = [line for line in lines if line.strip()]
                        texts.extend(lines)
                    except (RuntimeError, UnicodeDecodeError) as e:
                        print(f"Could not decode file {filename}: {e}")
    return texts

def split_dataset(texts, train_size=0.8, val_size=0.1):
    random.shuffle(texts)
    total_size = len(texts)
    train_end = int(train_size * total_size)
    val_end = int((train_size + val_size) * total_size)
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts}),
        'validation': Dataset.from_dict({'text': val_texts}),
        'test': Dataset.from_dict({'text': test_texts})
    })
    
def tokenize_batch(examples, tokenizer, max_length=512):
    tokenized_examples = {
        'input_ids': [],
        'attention_mask': [],
        'special_tokens_mask': []
    }
    for text in examples['text']:
        tokenized = tokenizer(text, padding=True, truncation=True, max_length=max_length)
        tokenized_examples['input_ids'].append(tokenized['input_ids'])
        tokenized_examples['attention_mask'].append(tokenized['attention_mask'])
        tokenized_examples['special_tokens_mask'].append(tokenized['special_tokens_mask'])
    return tokenized_examples
    
def load_vocab_from_txt():
    vocab_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/vocab.txt")
    if os.path.exists(vocab_file_path):
        print(f"Loading vocabulary from {vocab_file_path}...")
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    else:
        token_freq_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/combined_token_frequencies.txt")
        print(f"{vocab_file_path} not found. Generating vocabulary from {token_freq_path}...")
        vocab = {}
        try:
            with open(token_freq_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    token, _ = line.strip().split(': ')
                    vocab[token] = idx
            with open(vocab_file_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False)
            print(f"Vocabulary saved to {vocab_file_path}")
        except Exception as e:
            print(f"Error reading token frequency file: {e}")
    return vocab