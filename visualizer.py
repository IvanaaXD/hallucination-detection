import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize():
    try:
        with open("word2vec_model.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: You must run train.py first to generate the model!")
        return

    W_in = data['W_in']
    idx2word = data['idx2word']

    n_words = 150
    vectors = W_in[:n_words]
    labels = [idx2word[i] for i in range(n_words)]

    print("Calculating t-SNE projection (this might take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    vectors_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], edgecolors='k', c='lightblue')

    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', alpha=0.8)

    plt.title("Word2Vec Embeddings Visualization (t-SNE)")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    visualize()