import numpy as np
from numpy import einsum
import torch

class Encoder:
    def __init__(self, text, vocab_size, d_model, n_heads, d_ff, n_layers, dropout):
        self.text = text
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout

        self.__init_params()
        
    def __init_params(self):   
        # hidden weights in FP32
        d_k = d_v = self.d_model // self.n_heads
        # n_layers x n_heads x d_model x d_k
        self.Wk = torch.tensor(np.random.randn(self.n_layers, self.n_heads, self.d_model, d_k) * np.sqrt(2 / (self.d_model + self.d_model)), dtype=torch.float32)
        self.Wq = torch.tensor(np.random.randn(self.n_layers, self.n_heads, self.d_model, d_k) * np.sqrt(2 / (self.d_model + self.d_model)), dtype=torch.float32)
        self.Wv = torch.tensor(np.random.randn(self.n_layers, self.n_heads, self.d_model, d_v) * np.sqrt(2 / (self.d_model + self.d_model)), dtype=torch.float32)
        self.Wo = torch.tensor(np.random.randn(self.n_layers, self.d_model, self.d_model) * np.sqrt(2 / (self.d_model + self.d_model)), dtype=torch.float32)
        self.W1 = torch.tensor(np.random.randn(self.n_layers, self.d_model, self.d_ff) * np.sqrt(2 / (self.d_model + self.d_ff)), dtype=torch.float32)
        self.W2 = torch.tensor(np.random.randn(self.n_layers, self.d_ff, self.d_model) * np.sqrt(2 / (self.d_ff + self.d_model)), dtype=torch.float32)
        self.b1 = torch.tensor(np.random.randn(self.n_layers, self.d_ff, 1) * np.sqrt(2 / (self.d_model + self.d_ff)), dtype=torch.float32)
        self.b2 = torch.tensor(np.random.randn(self.n_layers, self.d_model, 1) * np.sqrt(2 / (self.d_ff + self.d_model)), dtype=torch.float32)
    
    # encoder steps
    # 1) tokenize text
    # 2) create embedding for each token
    # 2.1) add positional encoding
    # 3) pass through n_layers of encoder
    # 3.1) multi-head attention
    # 3.2) feed forward network
    # 4) return encoded text seq_len x d_model
    
    def forward(self):
        print("Forward pass:")
        # 1) tokenize text
        tokens = self.tokenize(self.text)
        # 2) create embedding for each token
        embeddings = self.embedding(tokens)
        # 3) pass through n_layers of encoder
        encoded_text = self.encoder(embeddings)
        return encoded_text
    
    def tokenize(self, text):
        # tokenize text
        return text.split()
    
    def embedding(self, tokens):
        # create embedding for each token
        return torch.rand(len(tokens), self.d_model)
    
    def encoder(self, embeddings):
        # pass through n_layers of encoder
        for i in range(self.n_layers):
            print(f"  Layer {i} ")
            embeddings = self.encoder_layer(embeddings, 
                                            self.Wk[i], self.Wq[i], self.Wv[i], self.Wo[i],
                                            self.W1[i], self.b1[i], self.W2[i], self.b2[i])
        return embeddings
    
    def encoder_layer(self, embeddings, Wk, Wq, Wv, Wo, W1, b1, W2, b2):
        # multi-head attention w/residual connection
        embeddings = self.multi_head_attention(embeddings, Wk, Wq, Wv, Wo)
        embeddings = embeddings + self.layer_norm(embeddings)
        # feed forward network w/residual connection
        embeddings = self.feed_forward_network(embeddings, W1, b1, W2, b2)
        embeddings = embeddings + self.layer_norm(embeddings)
        return embeddings
    
    def multi_head_attention(self, embeddings, Wk, Wq, Wv, Wo):
        # multi-head attention
        # separate embeddings into n_heads
        h = [self.attention(embeddings, Wk[i], Wq[i], Wv[i]) for i in range(self.n_heads)]

        # concatenate heads
        embeddings = torch.einsum('ik,kl->il', torch.cat(h, dim=1), Wo)
        
        return embeddings
    
    # parallelize attention computation
    # A(K,Q,V) = softmax(QK^T / sqrt(d_k))V
    # K = X * Wk
    # Q = X * Wq
    # V = X * Wv
    def attention(self, embeddings, Wk, Wq, Wv):
        K = torch.einsum('ik,kl->li', embeddings, Wk)
        Q = torch.einsum('ik,kl->il', embeddings, Wq)
        V = torch.einsum('ik,kl->il', embeddings, Wv)
        embeddings = torch.einsum('ik,kl->il', torch.softmax(
            torch.einsum('ik,kl->il',Q,K) / torch.sqrt(torch.tensor(self.d_model)), dim=1),V)
        return embeddings
    
    def feed_forward_network(self, embeddings, W1, b1, W2, b2):
        # feed forward network
        # FFN(X) = ReLU(XW1 + b1)W2 + b2
        # broadcast bias to match shape
        embeddings = torch.relu(torch.einsum('ik,kl->il', embeddings, W1) + b1.T)
        embeddings = torch.einsum('ik,kl->il', embeddings, W2) + b2.T
        return embeddings
    
    def layer_norm(self, embeddings):
        # layer normalization
        embeddings = (embeddings - torch.mean(embeddings)) / torch.std(embeddings)
        return embeddings

def main():
    Encoder("Hello, World!", 100, 512, 8, 2048, 6, 0.1).forward()

main()