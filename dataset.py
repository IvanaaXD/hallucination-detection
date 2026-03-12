import numpy as np
from collections import Counter
import random
import string

class Word2VecDataset:
    def __init__(self, text, window_size=2, num_neg_samples=5, subsample_threshold=1e-3):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        
        print("Cleaning and tokenizing text...")
        text = text.lower()
        
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator)
        
        raw_words = clean_text.split()
        
        print("Building vocabulary...")
        word_counts = Counter(raw_words)
        self.vocab = list(word_counts.keys())
        self.vocab_size = len(self.vocab)
        
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        total_words = len(raw_words)
        word_freqs = np.array([word_counts[word] / total_words for word in self.vocab])
        
        print("Performing subsampling...")
        drop_probs = 1 - np.sqrt(subsample_threshold / word_freqs)
        
        self.train_words = []
        for word in raw_words:
            word_idx = self.word2idx[word]
            if random.random() > drop_probs[word_idx]:
                self.train_words.append(word_idx)
                
        print(f"Kept {len(self.train_words)} out of {total_words} words after subsampling.")

        print("Initializing probabilities for negative sampling...")
        pow_freqs = word_freqs ** 0.75
        self.neg_sample_probs = pow_freqs / np.sum(pow_freqs)

    def get_negative_samples(self, current_context_indices):
        """
        Selects negative samples that are not in the current context.
        """
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            sampled_idx = np.random.choice(self.vocab_size, p=self.neg_sample_probs)
            if sampled_idx not in current_context_indices:
                neg_samples.append(sampled_idx)
        return neg_samples

    def generate_training_data(self):
        """
        Generator that yields (center_word_idx, context_word_idx, negative_samples_indices)
        """
        for i, center_word_idx in enumerate(self.train_words):
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(self.train_words), i + self.window_size + 1)
            
            context_indices = [self.train_words[j] for j in range(start_idx, end_idx) if j != i]
            
            for context_word_idx in context_indices:
                neg_samples = self.get_negative_samples(context_indices + [center_word_idx])
                
                yield center_word_idx, context_word_idx, neg_samples

# if __name__ == "__main__":
#     sample_text = """
#     Machine learning is a field of inquiry devoted to understanding and building methods that learn.
#     It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, 
#     known as training data, in order to make predictions or decisions without being explicitly programmed to do so.
#     """
    
#     print("=== RUNNING DATASET ===\n")
    
#     dataset = Word2VecDataset(sample_text, window_size=2, num_neg_samples=3)
    
#     print(f"\nVocabulary size: {dataset.vocab_size}")
#     print(f"First 10 words in vocabulary: {list(dataset.word2idx.keys())[:10]}")
#     print("-" * 40)
    
#     print("\nGenerating pairs (center, context, negative samples)...\n")
#     training_data_generator = dataset.generate_training_data()
    
#     for i in range(5):
#         try:
#             center_idx, context_idx, negative_indices = next(training_data_generator)
            
#             center_word = dataset.idx2word[center_idx]
#             context_word = dataset.idx2word[context_idx]
#             negative_words = [dataset.idx2word[n] for n in negative_indices]
            
#             print(f"Step {i+1}:")
#             print(f"  Center word (Input):  '{center_word}' (ID: {center_idx})")
#             print(f"  Context word (Positive): '{context_word}' (ID: {context_idx})")
#             print(f"  Negative words:         {negative_words} (IDs: {negative_indices})\n")
            
#         except StopIteration:
#             print("No more training data available.")
#             break