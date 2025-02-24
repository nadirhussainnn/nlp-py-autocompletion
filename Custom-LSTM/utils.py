"""
@author: nadirhussainnn
@date: 2025-02-13
@description: This file contains the utility functions for the code completion model.
"""
import torch
import pickle
import json
import re
import ast
import os


"""
@description: This function splits large tokens by common delimiters and camel case.
@param: token: The token to split.
@return: The split tokens.
"""
def split_large_token(token):
    """
    Break down large tokens by common delimiters and camel case.
    """
    # Split by common delimiters
    sub_tokens = re.split(r'[_\.]', token)
    
    # Further split camelCase tokens
    final_tokens = []
    for sub_token in sub_tokens:
        # Split by camel case boundaries
        camel_case_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', sub_token)
        final_tokens.extend(camel_case_split)
    
    # Remove empty tokens
    return [t for t in final_tokens if t]

"""
@description: This function tokenizes the code snippet.
@param: code_snippet: The code snippet to tokenize.
@return: The tokenized code snippet.
"""
def tokenize_code(code_snippet):

    token_pattern = re.compile(r'\w+|[^\w\s]')
    tokens = token_pattern.findall(code_snippet)

    # Handle long tokens by splitting
    final_tokens = []
    for token in tokens:
        if len(token) > 30:
            final_tokens.extend(split_large_token(token))
        else:
            final_tokens.append(token)
    
    return final_tokens

"""
@description: This function gets the device.
@return: The device.
"""
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

"""
@description: This function saves the vocabulary.
@param: vocab: The vocabulary to save.
@param: json_file: The file to save the vocabulary.
"""
def save_vocab(vocab, json_file="vocab.json"):
    with open(json_file, 'w') as json_out:
        json.dump(vocab, json_out, indent=4)
    print(f"Vocab saved successfully in {json_file}")

"""
@description: This function reads the files.
@param: base_dir: The base directory of the files.
@param: file_list: The file list to read.
@param: NUM_FILES: The number of files to read.
@return: The code snippet.
"""
def read_files(base_dir, file_list, NUM_FILES):
    code_snippet = ""
    file_count = 0

    try:
        with open(file_list, 'r') as f:
            for line in f:
                if file_count >= NUM_FILES:
                    break
                file_path = os.path.join(base_dir, line.strip())
                print(f"Reading file: {file_path}")
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                            code = code_file.read()
                            # Ensure valid Python syntax and consistent indentation
                            ast.parse(code)  # Try parsing the file; will raise error if invalid
                            code_snippet += code + "\n"
                            file_count += 1
                    except (SyntaxError, TabError) as e:
                        print(f"Skipping file due to parsing error: {file_path} ({e})")
                else:
                    print(f"File not found: {file_path}")

        print(f"Read {file_count} files successfully.")

    except FileNotFoundError:
        print(f"File {file_list} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return code_snippet

"""
@description: This function saves the configuration.
@param: model_config: The model configuration.
@param: vocab: The vocabulary.
@param: tokens: The tokens.
@param: config_file: The configuration file.
"""
def save_config(model_config, vocab, tokens, config_file="config.json"):
    config_data = {
        "model_config": model_config,
        "vocab_file": "vocab.pkl",
        "best_model_file": "best_model.pth"
    }

    # Save vocab and tokens
    with open("vocab.pkl", "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    # Save config JSON
    with open(config_file, "w") as config_json:
        json.dump(config_data, config_json)

"""
@description: This function generates the code completion.
@param: raw_code_snippet: The raw code snippet to generate the code completion.
@param: model: The model to generate the code completion.
@param: vocab: The vocabulary to generate the code completion.
@param: idx_to_token: The index to token mapping.
@param: device: The device to generate the code completion.
"""
def generate_code_completion(raw_code_snippet, model, vocab, idx_to_token, device, max_tokens=10, temperature=1.0):
    model.eval()
    
    # Tokenize and filter valid tokens
    seed_tokens = tokenize_code(raw_code_snippet.strip())
    seed_tokens = [token for token in seed_tokens if token in vocab]  

    if not seed_tokens:
        return "No valid tokens for inference."

    generated_tokens = seed_tokens[:]  

    for _ in range(max_tokens):
        input_ids = torch.tensor([[vocab[token] for token in generated_tokens]], device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0, -1] / temperature  
            probabilities = torch.softmax(logits, dim=-1)
            
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            next_token = idx_to_token[next_token_id]

         # Avoid repeated or redundant completions
        if next_token == generated_tokens[-1]:
            continue
        
        generated_tokens.append(next_token)


    return " ".join(generated_tokens)
