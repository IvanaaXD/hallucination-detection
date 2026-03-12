import numpy as np

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        
        self.W_out = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        
    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))
        
    def train_step(self, center_idx, context_idx, negative_indices, learning_rate):
        """
        Executes one optimization step: Forward pass -> Error calculation -> Weight update
        """
        v_c = self.W_in[center_idx]          
        u_p = self.W_out[context_idx]        
        U_n = self.W_out[negative_indices]   
        
        score_pos = self._sigmoid(np.dot(v_c, u_p))
        scores_neg = self._sigmoid(np.dot(U_n, v_c)) 
        
        err_pos = score_pos - 1.0 
        err_neg = scores_neg     
        
        grad_v_c = err_pos * u_p + np.dot(err_neg, U_n)
        
        self.W_out[context_idx] -= learning_rate * err_pos * v_c
        
        self.W_out[negative_indices] -= learning_rate * np.outer(err_neg, v_c)
        
        self.W_in[center_idx] -= learning_rate * grad_v_c
        
        return