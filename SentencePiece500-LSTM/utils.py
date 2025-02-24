"""
@author: nadirhussainnn
@date: 2025-02-13
@description: This file contains the utility functions for the code completion model.
"""

import torch
import sentencepiece as spm
import json
import os
import ast

"""
@description: This function trains the sentencepiece model.
@param: code_file: The file containing the code to train the sentencepiece model.
@param: model_prefix: The prefix of the sentencepiece model.
@param: vocab_size: The size of the vocabulary.
"""
def train_sentencepiece(code_file, model_prefix='spm_tokenizer', vocab_size=130000):
    spm.SentencePieceTrainer.train(
        input=code_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe'
    )
    print(f"SentencePiece model trained and saved as '{model_prefix}.model'")

"""
@description: This function tokenizes the code snippet.
@param: code_snippet: The code snippet to tokenize.
@param: sp: The sentencepiece model.
@return: The tokenized code snippet.
"""
def tokenize_code(code_snippet, sp):
    return sp.encode_as_pieces(code_snippet)


"""
@description: This function gets the device to use for training.
@return: The device to use for training.
"""
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

"""
@description: This function saves the inference configuration we can use for inference.
@param: model: The model to save the configuration for.
@param: vocab_size: The size of the vocabulary.
@param: sp_model_file: The path to the sentencepiece model.
@param: config_file: The path to save the configuration.
"""
def save_inference_config(model, vocab_size, sp_model_file='spm_tokenizer.model', config_file='config.json'):
    config_data = {
        "model_architecture": {
            "vocab_size": vocab_size,
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_dim": model.lstm.hidden_size
        },
        "sp_model_file": sp_model_file,
        "trained_model_file": "best_model.pth"
    }

    with open(config_file, 'w') as config_out:
        json.dump(config_data, config_out, indent=4)
    print(f"Inference configuration saved to {config_file}")


"""
@description: This function generates the code completion.
@param: raw_code_snippet: The raw code snippet to generate the code completion for.
@param: model: The model to use for generating the code completion.
@param: sp: The sentencepiece model.
@param: device: The device to use for generating the code completion.
@param: max_tokens: The maximum number of tokens to be generated.
@param: temperature: The temperature for the model i.e how random the model is.
@param: top_k: The top k tokens to be considered for the next token.
"""
def generate_code_completion(raw_code_snippet, model, sp, device, max_tokens=5, temperature=0.5, top_k=5):
    model.eval()

    """ Tokenizing the raw code snippet """
    seed_tokens = sp.encode_as_pieces(raw_code_snippet.strip())

    if not seed_tokens:
        return "No valid tokens for inference."

    """ Initializing the generated tokens with the seed tokens. """
    generated_tokens = seed_tokens[:]

    """ Generating the code completion, upto max_tokens """
    for _ in range(max_tokens):

        """ Converting the generated tokens to a tensor """
        input_ids = torch.tensor(
            [[sp.piece_to_id(token) for token in generated_tokens]],
            device=device,
            dtype=torch.long
        )

        """ Generating the next token """
        with torch.no_grad():
            outputs = model(input_ids)

            """ Calculating the logits """
            logits = outputs[0, -1] / temperature

            """ Calculating the probabilities """
            probabilities = torch.softmax(logits, dim=-1)
            top_k_prob, top_k_indices = torch.topk(probabilities, top_k)
            
            """ Normalizing the probabilities """
            top_k_prob = top_k_prob / torch.sum(top_k_prob)  
            print("Top K Probabilities: ", top_k_prob)
            """ Sampling the next token """
            next_token_id = torch.multinomial(top_k_prob, num_samples=1).item()

            """ Converting the next token id to a token """
            next_token = sp.id_to_piece(top_k_indices[next_token_id].item())

            """ If the next token is the end of the sentence token, break """
            if next_token == '</s>':
                break

            """ Appending the next token to the generated tokens """
            generated_tokens.append(next_token)

    """ Decoding the generated tokens to a string """
    return sp.decode_pieces(generated_tokens)


"""
@description: This function reads the files from the given directory: data.
@param: base_dir: The base directory to read the files from.
@param: file_list: The file list  that contains names of files to be read.
@param: NUM_FILES: The number of files to read.
@return: The code snippet that contains the code from the files.
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
                            ast.parse(code) 
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