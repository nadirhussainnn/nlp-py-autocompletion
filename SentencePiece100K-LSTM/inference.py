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
    
    #  Test cases from train and val files


    test_cases = [
            # data/9miao/Firefly/firefly/dbentrust/memclient.py
            ("keys = [self.produceKey(key) for", "keys = [self.produceKey(key) for key"),

            # data/Akagi201/learning-python/dnslib/decode_packet.py
            ("d = DNSRecord.parse(", "d = DNSRecord.parse("),
            
            # data/0rpc/zerorpc-python/zerorpc/events.py
            ("if zmq_socket_type in (zmq.PULL,", "if zmq_socket_type in (zmq.PULL, zmq.SUB"), 
            ("for endpoint_ in self._resolve_endpoint(", "for endpoint_ in self._resolve_endpoint(endpoint"),

            #data/HenryHu/pybbs/Config.py
            ("SESSION_TIMEOUT =", "SESSION_TIMEOUT = datetime"),

            # data/HenryHu/pybbs/digest.py
            ("while (self.items[_id].EffectiveId(user) < ", "while (self.items[_id].EffectiveId(user) < target"),
            
            #data/HenryHu/pybbs/xmppserver.py
            ("def recv_close(self):\nreturn", "def recv_close(self):\nreturn self"),

            #Unknown
            ("import numpy", "import numpy as"),
            ("pd.read", "pd.read_")
    ]

    # Run each test case
    for raw_code, expected in test_cases:
        print(f"\nInput: {raw_code}")
        completion = generate_code_completion(raw_code, model, sp, device)
        print("Generated Completion:", completion)
        print("Expected Completion:", expected)
        print("Pass:", completion.startswith(expected))



if __name__ == "__main__":
    inference()