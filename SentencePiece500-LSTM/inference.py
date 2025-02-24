"""
@author: nadirhussainnn
@date: 2025-02-13
@description: This file contains the code for the code completion model inference.
"""

import torch
import sentencepiece as spm
import json
from train import CodeCompletionModel, get_device, generate_code_completion
from utils import *

"""
@description: This function is the main function for inference.
"""
def inference():

    """ Loading the inference configuration """
    with open("config.json", "r") as config_in:
        config_data = json.load(config_in)

    """ Creating the model from the configuration """
    model = CodeCompletionModel(config_data["model_architecture"]["vocab_size"], config_data["model_architecture"]["embedding_dim"], config_data["model_architecture"]["hidden_dim"])

    device = get_device()
    
    """ Loading the trained model """
    model.load_state_dict(torch.load(config_data["trained_model_file"], map_location=torch.device(device)))

    """ Setting the model to evaluation mode """
    model.eval()
    model.to(device)
    
    """ Creating instance of the sentencepiece model from the configuration """
    sp = spm.SentencePieceProcessor(model_file=config_data["sp_model_file"])
    
    raw_code = "if __name__ == "
    
    completion = generate_code_completion(raw_code, model, sp, device)
    print("Generated Completion:", completion)

if __name__ == "__main__":
    inference()