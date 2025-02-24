"""
@author: nadirhussainnn
@date: 2025-02-13
@description: This file contains the utility functions for the code completion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import tokenize_code, get_device, save_vocab, read_files, save_config
import matplotlib.pyplot as plt
import numpy as np

"""
@description: This class is used to create a code completion model.
@param: vocab_size: The size of the vocabulary.
@param: embedding_dim: The embedding dimension.
@param: hidden_dim: The hidden dimension.
@return: The code completion model.
"""
class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(CodeCompletionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


"""
@description: This class is used to create a dataset of code snippets.
@param: tokens: The tokens of the code snippets.
@param: seq_length: The sequence length of the code snippets.
@return: The dataset of code snippets.
"""
class CodeDataset(Dataset):
    def __init__(self, tokens, seq_length):
        self.tokens = tokens
        self.seq_length = seq_length
        self.vocab = {token: idx for idx, token in enumerate(set(tokens))}
        print("self.vocab", self.vocab)
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        print("self.idx_to_token", self.idx_to_token)

        self.data = [
            (self.tokens[i:i + seq_length], self.tokens[i + 1:i + seq_length + 1])
            for i in range(len(self.tokens) - seq_length)
        ]
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_ids = [self.vocab[token] for token in input_seq]
        target_ids = [self.vocab[token] for token in target_seq]
        return torch.tensor(input_ids), torch.tensor(target_ids)


"""
@description: This function is used to train the model.
@param: model: The model to train.
@param: dataset: The dataset to train on.
@param: num_epochs: The number of epochs to train on.
@param: batch_size: The batch size.
@param: patience: The patience for early stopping.
@param: model_save_path: The path to save the model.
"""
def train_model(model, dataset, num_epochs=10, batch_size=32, patience=5, model_save_path="best_model.pth"):
    device = get_device()
    print(f"Using device: {device}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        correct_tokens = 0
        total_tokens = 0


        # Training loop
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            # Exp: outputs[[[get user -balance-]]]
            # targets: [[get user -balance- ]]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            """ Calculating the accuracy """
            _, predicted = torch.max(outputs, dim=-1)
            total_tokens += targets.numel()
            correct_tokens += (predicted == targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracy = 100 * correct_tokens / total_tokens

        # Calculate Perplexity
        train_perplexity = np.exp(train_loss)
        val_perplexity = np.exp(val_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.2f}, Val Perplexity: {val_perplexity:.2f}")

        # Save the best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")
        else:
            early_stop_counter += 1

        # Early stopping condition
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plotting and saving loss graph after every epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_epoch_{epoch + 1}.png')
    plt.close()

    # Plotting and saving perplexity graph after every epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_perplexities) + 1), train_perplexities, label='Train Perplexity')
    plt.plot(range(1, len(val_perplexities) + 1), val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'perplexity_plot_epoch_{epoch + 1}.png')
    plt.close()

    print("Training complete. Loss plot saved as 'loss_plot.png'.")



if __name__ == "__main__":

    base_dir = ""
    file_list = "py150_files/python100k_train.txt"
    
    NUM_FILES = 500
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    PATIENCE = 3
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    SEQ_LENGTH = 20
    
    code_snippet = read_files(base_dir, file_list, NUM_FILES)
    print(f"Loaded {len(code_snippet)} characters of code")

    # Tokenization  
    tokens = tokenize_code(code_snippet)    
    print(f"Tokenized {len(tokens)} tokens")
    
    dataset = CodeDataset(tokens, SEQ_LENGTH)
    print(f"Dataset created with {len(dataset)} sequences")


    vocab_size = len(dataset.vocab)
    device = get_device()
    save_vocab(dataset.vocab)


    model = CodeCompletionModel(vocab_size).to(device)
    print(f"Model created with {vocab_size} vocabulary size")
    # Save the config, tokens, and vocab
    model_config = {"vocab_size": vocab_size, "embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM}
    save_config(model_config, dataset.vocab, tokens)
    print("Configuration, vocabulary, and tokens saved successfully.")

    # Train Model
    train_model(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
    print("Training completed")