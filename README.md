# HALLUCINATION DETECTION

# Word2Vec in Pure NumPy (Skip-gram with Negative Sampling)

This repository contains a from-scratch, pure NumPy implementation of the Word2Vec algorithm, specifically the **Skip-gram model with Negative Sampling (SGNS)**. 

This project was developed without the use of high-level machine learning frameworks (like PyTorch or TensorFlow) to demonstrate a deep understanding of the underlying mathematics, forward propagation, loss calculation, and backpropagation using pure linear algebra.

## Features
* **Pure NumPy Architecture:** All matrix multiplications and gradient updates are implemented using vectorized NumPy operations for optimal performance.
* **Negative Sampling:** Efficiently approximates the Softmax denominator by sampling negative context words based on their unigram distribution ($P(w_i) \propto f(w_i)^{3/4}$).
* **Frequent Word Subsampling:** Implements Mikolov's subsampling strategy to discard highly frequent words (e.g., "the", "a") to speed up training and improve representations of rare words.
* **Learning Rate Decay:** Linear learning rate scheduling ensures smooth convergence.
* **t-SNE Visualization:** Includes a script to project the high-dimensional word embeddings into a 2D space for visual evaluation of semantic clustering.
* **CLI Interface:** Configurable hyperparameters via `argparse`.

## Project Structure
* `dataset.py`: Handles text tokenization, vocabulary building, subsampling, and generating (center, context, negative) training triplets.
* `model.py`: Contains the `Word2VecSGNS` class with the weight matrices ($W_{in}$, $W_{out}$) and the core optimization step (forward pass, gradients, SGD parameter updates).
* `train.py`: The main training loop. Automatically downloads a sample dataset (e.g., Tiny Shakespeare) if not provided, trains the model, and evaluates it using Cosine Similarity.
* `visualize.py`: Loads the trained model weights and plots a 2D t-SNE visualization of the word embeddings.

## Setup and Installation

### 1. Create a Virtual Environment
It is recommended to run this project in an isolated Python environment.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate

```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 2. Install Dependencies

The core algorithm only requires `numpy`. Visualization requires `matplotlib` and `scikit-learn`. Install them using pip:

```bash
pip install numpy matplotlib scikit-learn

```

## How to Run

### Training the Model

You can start the training process using the default parameters. The script will automatically download a sample dataset (Tiny Shakespeare) and begin training.

```bash
python train.py

```

**Customizing Hyperparameters:**
The training script supports various command-line arguments:

```bash
python train.py --epochs 5 --lr 0.025 --dim 50 --window 3 --neg 5

```

* `--epochs`: Number of training iterations over the dataset (default: 5).
* `--lr`: Initial learning rate for Stochastic Gradient Descent (default: 0.025).
* `--dim`: Dimensionality of the word embeddings (default: 50).
* `--window`: Context window size (default: 3).
* `--neg`: Number of negative samples per positive pair (default: 5).

Once training is complete, the script saves the model weights and vocabulary mappings to `word2vec_model.pkl`.

### Visualizing the Embeddings

After successfully training the model and generating the `.pkl` file, you can visualize the learned semantic relationships:

```bash
python visualize.py

```

This will open a Matplotlib window displaying a 2D t-SNE scatter plot of the most frequent words in the vocabulary, showcasing their semantic clustering.

```
