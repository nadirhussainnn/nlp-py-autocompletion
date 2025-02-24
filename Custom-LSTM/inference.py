import torch
import json
import pickle
from utils import get_device, generate_code_completion
from train import CodeCompletionModel

def load_config(config_file="config.json"):
    """Load model configuration, vocabulary, and tokens."""
    with open(config_file, "r") as config_json:
        config_data = json.load(config_json)

    with open(config_data["vocab_file"], "rb") as vocab_file:
        vocab = pickle.load(vocab_file)

    print("Configuration and vocabulary loaded successfully.")
    return config_data, vocab


def load_model(config_data, device):
    """Load the trained model with saved configuration."""
    vocab_size = config_data["model_config"]["vocab_size"]
    embedding_dim = config_data["model_config"]["embedding_dim"]
    hidden_dim = config_data["model_config"]["hidden_dim"]

    model = CodeCompletionModel(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(config_data["best_model_file"], map_location=device))
    model.eval()

    print("Model loaded successfully.")
    return model


if __name__ == "__main__":

    # Load configuration and vocabulary
    config_file = "config.json"
    config_data, vocab = load_config(config_file)
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # Load the trained model
    device = get_device()
    model = load_model(config_data, device)

    # Test inference
    raw_code = "if zmq_socket_type in (zmq.PULL,"

    test_cases = [
            # data/9miao/Firefly/firefly/dbentrust/memclient.py
            ("keys = [self.produceKey(key) for", "keys = [self.produceKey(key) for key in keys]"),

            # data/Akagi201/learning-python/dnslib/decode_packet.py
            ("d = DNSRecord.parse(", "d = DNSRecord.parse(packet)"),
            
            # data/0rpc/zerorpc-python/zerorpc/events.py
            ("if zmq_socket_type in (zmq.PULL,", "if zmq_socket_type in (zmq.PULL, zmq.SUB, zmq.DEALER, zmq.ROUTER):"), 
            ("for endpoint_ in self._resolve_endpoint(", "for endpoint_ in self._resolve_endpoint(endpoint, resolve):"),

            #data/HenryHu/pybbs/Config.py
            ("SESSION_TIMEOUT =", "SESSION_TIMEOUT = datetime.timedelta(30)"),

            # data/HenryHu/pybbs/digest.py
            ("while (self.items[_id].EffectiveId(user) < ", "while (self.items[_id].EffectiveId(user) < target):"),
            
            #data/HenryHu/pybbs/xmppserver.py
            ("def recv_close(self):\n return ", "def recv_close(self):\nreturn self.close()"),

            #Unknown
            ("import numpy", "import numpy as np"),
            ("pd.read", "pd.read_csv()")
    ]

    # Run each test case
    for raw_code, expected in test_cases:
        print(f"\nInput: {raw_code}")
        completion = generate_code_completion(raw_code, model, vocab, idx_to_token, device)
        print("Generated Completion:", completion)
        print("Expected Completion:", expected)
        print("Pass:", completion.startswith(expected))
