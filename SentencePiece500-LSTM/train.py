"""
@author: nadirhussainnn
@date: 2025-02-13
@description: This file contains the code for the code completion model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import matplotlib.pyplot as plt
from utils import *
import numpy as np


"""
@description: This class is a dataset for the code completion model.
@param: tokens: The tokens of the code snippet.
@param: seq_length: The sequence length of the code snippet.
@param: sp_model: The sentencepiece model.
"""
class CodeDataset(Dataset):
    def __init__(self, tokens, seq_length, sp_model="spm_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)
        self.tokens = tokens
        self.seq_length = seq_length

        self.vocab_size = self.sp.vocab_size()
        self.idx_to_token = {i: self.sp.id_to_piece(i) for i in range(self.vocab_size)}
        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}

        self.data = [
            (self.tokens[i:i + seq_length], self.tokens[i + 1:i + seq_length + 1])
            for i in range(len(tokens) - seq_length)
        ]

    """
    @description: This function returns the length of the dataset.
    @return: The length of the dataset.
    """
    def __len__(self):
        return len(self.data)
    
    """
    @description: This function returns the item at the given index.
    @param: idx: The index of the item to return.
    @return: The item at the given index as a tensor.
    """
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_ids = [self.token_to_idx.get(token, self.sp.unk_id()) for token in input_seq]
        target_ids = [self.token_to_idx.get(token, self.sp.unk_id()) for token in target_seq]
        return torch.tensor(input_ids), torch.tensor(target_ids)


"""
@description: This class is the code completion model.
@param: vocab_size: The size of the vocabulary.
@param: embedding_dim: The dimension of the embedding.
@param: hidden_dim: The dimension of the hidden layer.
"""
class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(CodeCompletionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    """
    @description: This function is the forward pass of the model.
    @param: x: The input to the model.
    @return: The output of the model.
    """
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


"""
@description: This function trains the model.
@param: model: The model to train.
@param: dataset: The dataset to train the model on.
@param: num_epochs: The number of epochs to train the model.
@param: batch_size: The batch size to train the model on.
@param: patience: The patience for early stopping.
@param: model_save_path: The path to save the best model.
"""
def train_model(model, dataset, num_epochs=10, batch_size=32, patience=5, model_save_path="best_model.pth"):
    device = get_device()
    print(f"Using device: {device}")

    """ Splitting the dataset into training and validation sets """
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
        total_tokens = 0
        correct_tokens = 0

        """ Training loop """
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            """ Calculating the accuracy """
            _, predicted = torch.max(outputs, dim=-1)
            total_tokens += targets.numel()
            correct_tokens += (predicted == targets).sum().item()

            """ Calculate the perplexity """
            train_perplexity = np.exp(train_loss)
            val_perplexity = np.exp(val_loss)
            train_perplexities.append(train_perplexity)
            val_perplexities.append(val_perplexity)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        """ Validation loop """
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                val_loss += loss.item()

        """ Calculate the average loss for the epoch """
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        accuracy = 100 * correct_tokens / total_tokens
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.2f}, Val Perplexity: {val_perplexity:.2f}")

        """ Save the best model checkpoint """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")
        else:
            early_stop_counter += 1

        """ Early stopping condition """
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    """ Plotting the training and validation loss graph """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

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


"""
@description: This function is the main function for training the model.
"""
def main():

    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    PATIENCE = 3
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    SEQ_LENGTH = 50

    """ Training the sentencepiece model using the collected code that contains code from first 500 files """
    train_sentencepiece("collected_code.py")
    
    """ Tokenizing the code snippet using the sentencepiece model """
    sp = spm.SentencePieceProcessor(model_file='spm_tokenizer.model')
    
    """ Reading the collected code and tokenizing it, that is to be used for training """
    code_snippet = open("collected_code.py", "r").read()
    tokens = tokenize_code(code_snippet, sp)

    """ Creating the dataset using the tokenized code """
    dataset = CodeDataset(tokens, SEQ_LENGTH)
    vocab_size = dataset.vocab_size
    device = get_device()

    """ Creating the model """
    model = CodeCompletionModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)

    """ Saving the inference configuration """
    save_inference_config(model, vocab_size)

    """ Training the model """
    train_model(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
    print("Training completed")

    """ Saving the model """



if __name__ == "__main__":

    """ Training the model """
    main()