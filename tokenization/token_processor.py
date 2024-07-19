import os
from collections import Counter

class TokenProcessor:
    """
    A class to process token frequencies from multiple files and perform various operations 
    on the token data, including combining frequencies, cleaning, and saving the results.
    """

    def __init__(self, base_dir):
        """
        Initialize the TokenProcessor with the base directory containing the tokenized datasets.
        
        Args:
            base_dir (str): The base directory path where tokenized dataset folders are located.
        """
        self.base_dir = base_dir
        self.folders = [
            os.path.join(base_dir, "tokenized_datasets_folder1"),
            os.path.join(base_dir, "tokenized_datasets_folder2"),
            os.path.join(base_dir, "tokenized_datasets_folder3"),
            os.path.join(base_dir, "tokenized_datasets_folder4"),
            os.path.join(base_dir, "tokenized_datasets_folder5")
        ]
        self.combined_counter = Counter()
        self.combined_token_file_counter = Counter()

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
                    elif line[ch] == ",":
                        final = ch
                    elif line[ch] == ":":
                        start = ch + 2
                        break
                token, freq = line[:start - 2], line[start:final]
                token_counter[token] = int(freq)
                token_file_counter[token] = int(line[start1:final1])
        return token_counter, token_file_counter

    def process_folders(self):
        """
        Process each folder to read token frequencies and update the combined counters.
        """
        for folder in self.folders:
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

    def find_non_chinese_or_english_chars(self, characters):
        """
        Find characters in a list that are neither Chinese nor English characters.

        Args:
            characters (list of str): A list of characters.

        Returns:
            list: A list of characters that are neither Chinese nor English.
        """
        non_chinese_or_english_chars = [ch for ch in characters if not self.is_chinese_char(ch) and not self.is_english_char(ch)]
        return non_chinese_or_english_chars

    def clean_combined_counter(self):
        """
        Clean the combined counter by removing tokens based on specific criteria:
        1. Tokens with a frequency less than or equal to 50.
        2. Tokens longer than 1 character that contain non-Chinese characters.
        3. Tokens longer than 1 character that contain digits.
        4. Tokens that appear in fewer than 15 documents.
        5. Remove the tokens that meet the above criteria from the combined counter and add single characters not in the combined counter.
        """
        delet = []

        # 1. Remove tokens with frequency <= 50
        for i in self.combined_counter:
            if self.combined_counter[i] <= 50:
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

        # 4. Remove tokens appearing in fewer than 15 documents
        for i in self.combined_token_file_counter:
            if self.combined_token_file_counter[i] < 15:
                delet.append(i)

        # 5. Remove the tokens that meet the above criteria and add single characters not in the combined counter
        for i in delet:
            for j in i:
                if j not in self.combined_counter:
                    self.combined_counter[j] = 1
            del self.combined_counter[i]

    def save_combined_counter(self, output_file):
        """
        Save the combined token frequencies to a specified file.

        Args:
            output_file (str): The path to the output file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for token, freq in self.combined_counter.items():
                f.write(f"{token}: {freq}\n")

        print(f"Combined token frequencies saved to {output_file}")
