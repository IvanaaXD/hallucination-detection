import numpy as np
import time
import argparse
import os
import urllib.request
from dataset import Word2VecDataset
from model import Word2VecSGNS
import pickle

def download_sample_data(file_path):
    """Downloads the Shakespeare dataset if the file doesn't exist locally."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Downloading dataset from the internet...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, file_path)
        print("Download completed successfully!\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def get_similar_words(word, dataset, model, top_n=5):
    if word not in dataset.word2idx:
        return f"Word '{word}' is not in the vocabulary."
    
    word_idx = dataset.word2idx[word]
    word_vec = model.W_in[word_idx] 
    
    similarities = []
    for i in range(dataset.vocab_size):
        if i == word_idx:
            continue 
        
        sim = cosine_similarity(word_vec, model.W_in[i])
        similarities.append((dataset.idx2word[i], sim))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def train(file_path, epochs, lr, embedding_dim, window_size, num_neg_samples):

    print("1. Preparing text...")
    text = download_sample_data(file_path)
    print(f"Loaded text of length {len(text)} characters.")
    
    print("\n2. Initializing data...")
    dataset = Word2VecDataset(text, window_size=window_size, num_neg_samples=num_neg_samples)
    
    print(f"\n3. Initializing model (Dimensions: {embedding_dim})...")
    model = Word2VecSGNS(dataset.vocab_size, embedding_dim=embedding_dim)
    
    print(f"\n4. Starting training ({epochs} epochs, Initial LR: {lr})...")
    total_steps = len(dataset.train_words) * window_size * 2 * epochs
    current_step = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        data_generator = dataset.generate_training_data()
        
        for center_idx, context_idx, negative_indices in data_generator:
            current_lr = max(lr * (1 - current_step / total_steps), 0.0001)
            model.train_step(center_idx, context_idx, negative_indices, current_lr)
            current_step += 1
            
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch + 1}/{epochs} completed in {elapsed:.2f}s (LR: {current_lr:.5f})")
            
    print("\n5. Testing learned embeddings...")

    test_words = ['king', 'queen', 'love', 'death', 'romeo']
    for w in test_words:
        sims = get_similar_words(w, dataset, model)
        print(f"\nMost similar words for '{w}':")
        if isinstance(sims, str):
            print(sims)
        else:
            for sim_word, score in sims:
                print(f"  -> {sim_word} (similarity: {score:.3f})")

    print("\n6. Saving model...")
    model_data = {
        'W_in': model.W_in,
        'word2idx': dataset.word2idx,
        'idx2word': dataset.idx2word
    }
    with open("word2vec_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved as 'word2vec_model.pkl'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Word2Vec SGNS model in NumPy.")
    
    parser.add_argument("--file", type=str, default="shakespeare.txt", help="Path to the text file for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.025, help="Initial learning rate")
    parser.add_argument("--dim", type=int, default=50, help="Vector dimension (embedding size)")
    parser.add_argument("--window", type=int, default=3, help="Context window size")
    parser.add_argument("--neg", type=int, default=5, help="Number of negative samples per positive")
    
    args = parser.parse_args()
    
    train(file_path=args.file,
          epochs=args.epochs, 
          lr=args.lr, 
          embedding_dim=args.dim, 
          window_size=args.window, 
          num_neg_samples=args.neg)