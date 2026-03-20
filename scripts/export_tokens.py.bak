import os
import sys
import json
import pickle
import hashlib
import struct
import numpy as np

VOCAB_SIZE = 8192
TOKENIZER_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer", "tokenizer.pkl")
OUTPUT_DIR = os.path.join("public", "data")
TOKENS_BIN = os.path.join(OUTPUT_DIR, "tokens.bin")
META_JSON = os.path.join(OUTPUT_DIR, "tokens_meta.json")

SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
Neural networks are inspired by the structure of the human brain, consisting of interconnected nodes called neurons.
Deep learning uses multiple layers of neural networks to model complex patterns in large datasets.
Transformers have revolutionized natural language processing by enabling parallel processing of sequential data.
The attention mechanism allows models to focus on relevant parts of the input when generating each output token.
Large language models are trained on vast corpora of text to predict the next token in a sequence.
Tokenization is the process of converting raw text into a sequence of integers that a model can process.
Byte pair encoding is a common tokenization algorithm that iteratively merges the most frequent pairs of characters.
Word-level tokenization splits text on whitespace and punctuation, creating a vocabulary of individual words.
Character-level tokenization treats each character as a separate token, resulting in a small vocabulary but long sequences.
The vocabulary size is a hyperparameter that balances expressiveness with computational efficiency.
Gradient descent is an optimization algorithm used to minimize the loss function during training.
Backpropagation computes gradients of the loss with respect to each parameter in the network.
The learning rate controls how much the model weights are updated during each training step.
Batch normalization stabilizes training by normalizing the inputs to each layer.
Dropout is a regularization technique that randomly sets a fraction of activations to zero during training.
Overfitting occurs when a model learns the training data too well and fails to generalize to new data.
Cross-entropy loss is commonly used for classification tasks, measuring the difference between predicted and true distributions.
Perplexity is a measure of how well a language model predicts a sample, with lower values indicating better performance.
The softmax function converts raw logits into a probability distribution over the vocabulary.
Embedding layers map discrete token IDs to continuous vector representations in a high-dimensional space.
Positional encodings add information about token positions to the input embeddings since transformers lack inherent sequence order.
Multi-head attention allows the model to attend to information from different representation subspaces simultaneously.
Feed-forward networks in transformers consist of two linear transformations with a nonlinear activation function in between.
Layer normalization is applied before each sub-layer in modern transformer architectures, known as pre-norm.
Residual connections help gradients flow through deep networks by adding the input of each sub-layer to its output.
The key, query, and value projections in attention compute how much each token should attend to every other token.
Causal masking prevents tokens from attending to future positions during autoregressive generation.
KV caching stores previously computed key and value tensors to avoid redundant computation during inference.
Quantization reduces the precision of model weights and activations to decrease memory usage and increase inference speed.
Knowledge distillation trains a smaller student model to mimic the behavior of a larger teacher model.
Fine-tuning adapts a pre-trained model to a specific downstream task using a smaller labeled dataset.
Transfer learning leverages knowledge from one task to improve performance on a related task.
Data augmentation artificially increases the training set size by applying transformations to existing examples.
Learning rate scheduling adjusts the learning rate during training, often decreasing it over time.
Warmup gradually increases the learning rate from zero to the target value during the initial training steps.
Weight decay adds a penalty proportional to the magnitude of the weights to the loss function, encouraging simpler models.
Mixed precision training uses both 16-bit and 32-bit floating point numbers to speed up training while maintaining accuracy.
Distributed training splits the workload across multiple GPUs or machines to handle larger models and datasets.
Model parallelism divides the model itself across devices, while data parallelism distributes the data.
Pipeline parallelism assigns different layers to different devices, processing micro-batches in a pipelined fashion.
The transformer architecture consists of an encoder and decoder, each composed of stacked self-attention and feed-forward layers.
Encoder-only models like BERT are designed for understanding tasks such as classification and named entity recognition.
Decoder-only models like GPT are optimized for generation tasks, predicting the next token in an autoregressive fashion.
Encoder-decoder models like T5 are suited for sequence-to-sequence tasks such as translation and summarization.
Prompt engineering involves crafting input text that guides the model toward producing desired outputs without updating weights.
In-context learning allows large language models to adapt to new tasks based on examples provided in the prompt.
Chain-of-thought prompting encourages models to show their reasoning steps, often improving performance on complex tasks.
Retrieval-augmented generation combines a retriever that fetches relevant documents with a generator that produces answers.
The scaling laws for neural language models describe how performance improves with increases in model size, data, and compute.
Emergent abilities appear in large models that are not present in smaller ones, such as multi-step reasoning.
Alignment techniques like RLHF train models to follow human preferences and instructions more reliably.
Constitutional AI uses a set of principles to guide model behavior without requiring human feedback for every decision.
Safety measures include content filtering, output constraints, and red-teaming to identify and mitigate harmful behaviors.
Evaluation benchmarks test model capabilities across tasks like reasoning, knowledge, coding, and language understanding.
Few-shot evaluation measures how well a model can learn from a small number of examples provided in the context.
Zero-shot evaluation tests model performance on tasks without any task-specific examples in the prompt.
The HELM benchmark provides a comprehensive evaluation of language models across many dimensions and scenarios.
Human evaluation remains the gold standard for assessing the quality and usefulness of model outputs.
Automated metrics like BLEU, ROUGE, and BERTScore provide quick but imperfect estimates of output quality.
Perplexity on a held-out test set is a standard intrinsic evaluation metric for language models.
Bits per byte normalizes perplexity across different tokenizers, enabling fair comparison between models with different vocabularies.
The information content of a text determines the theoretical minimum number of bits needed to encode it.
Shannon entropy quantifies the average information content per symbol in a message source.
Natural language has significant redundancy, which is why compression algorithms can achieve high ratios on text data.
Zipf's law states that the frequency of a word is inversely proportional to its rank in the frequency table.
Heaps' law describes how the vocabulary size grows as a function of the corpus size.
Stop words are common words like the, is, and that carry little semantic meaning and are often removed in text processing.
Stemming reduces words to their root form, while lemmatization considers the word's context and part of speech.
Named entity recognition identifies and classifies named entities such as persons, organizations, and locations in text.
Part-of-speech tagging assigns grammatical categories to each word in a sentence.
Dependency parsing analyzes the grammatical structure of a sentence, identifying relationships between words.
Sentiment analysis determines the emotional tone of a text, classifying it as positive, negative, or neutral.
Text classification assigns predefined labels to documents based on their content.
Topic modeling discovers abstract topics that occur in a collection of documents.
Word embeddings represent words as dense vectors in a continuous space, capturing semantic relationships.
Word2Vec learns word embeddings by predicting surrounding words in a context window.
GloVe combines global matrix factorization with local context window methods to learn word vectors.
FastText extends Word2Vec by representing words as bags of character n-grams, handling out-of-vocabulary words.
Contextual embeddings from models like BERT generate different representations for the same word depending on its context.
Sentence embeddings aggregate word or token representations into a fixed-size vector representing the entire sentence.
Cosine similarity measures the angle between two vectors, commonly used to compare embeddings.
Dimensionality reduction techniques like PCA and t-SNE help visualize high-dimensional embedding spaces.
Clustering algorithms group similar embeddings together, useful for discovering patterns in unlabeled data.
Anomaly detection identifies data points that deviate significantly from the expected distribution.
Time series analysis examines data points collected over time to identify trends, seasonality, and patterns.
Recurrent neural networks process sequential data by maintaining a hidden state that captures information from previous steps.
Long short-term memory networks address the vanishing gradient problem in RNNs through gating mechanisms.
Gated recurrent units simplify the LSTM architecture by combining the forget and input gates into a single update gate.
Convolutional neural networks apply learnable filters to local regions of the input, capturing spatial hierarchies.
Pooling layers reduce the spatial dimensions of feature maps, providing translation invariance and reducing computation.
Batch normalization layers normalize activations within a mini-batch, reducing internal covariate shift.
Residual networks enable training of very deep architectures by introducing skip connections.
DenseNet connects each layer to every other layer in a feed-forward fashion, encouraging feature reuse.
MobileNet uses depthwise separable convolutions to reduce the number of parameters and computations.
EfficientNet scales network width, depth, and resolution uniformly using a compound scaling method.
Vision transformers apply the transformer architecture to image patches, achieving competitive performance on image classification.
Object detection identifies and localizes objects within an image, outputting bounding boxes and class labels.
Semantic segmentation assigns a class label to each pixel in an image.
Instance segmentation combines object detection and semantic segmentation, distinguishing between individual instances.
Generative adversarial networks consist of a generator that creates samples and a discriminator that evaluates them.
Variational autoencoders learn a latent representation of data by optimizing a reconstruction loss and a regularization term.
Normalizing flows transform a simple distribution into a complex one through a series of invertible mappings.
Diffusion models generate data by gradually denoising samples from a random noise distribution.
Energy-based models define an energy function that assigns low energy to desirable configurations and high energy to undesirable ones.
Contrastive learning learns representations by pulling similar pairs together and pushing dissimilar pairs apart in embedding space.
Self-supervised learning creates supervisory signals from the data itself, eliminating the need for manual labeling.
Meta-learning or learning to learn adapts learning algorithms to new tasks with minimal experience.
Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward in an environment.
Q-learning estimates the value of state-action pairs to derive an optimal policy without a model of the environment.
Policy gradient methods directly optimize the policy by estimating gradients of the expected reward.
Actor-critic methods combine value-based and policy-based approaches, using an actor to select actions and a critic to evaluate them.
Multi-agent reinforcement learning studies how multiple agents interact and learn in shared environments.
Sim-to-real transfer trains policies in simulation and transfers them to physical robots, bridging the reality gap.
Imitation learning trains agents to mimic expert behavior from demonstration data.
Inverse reinforcement learning infers the reward function from observed expert behavior.
Curriculum learning presents training examples in a meaningful order, gradually increasing difficulty.
Active learning selects the most informative unlabeled examples for labeling, reducing annotation costs.
Semi-supervised learning leverages both labeled and unlabeled data to improve model performance.
Few-shot learning aims to learn new concepts from only a handful of examples.
Zero-shot learning recognizes classes that were not seen during training by leveraging semantic descriptions.
Continual learning enables models to learn new tasks sequentially without forgetting previously learned knowledge.
Federated learning trains models across decentralized devices without sharing raw data, preserving privacy.
Differential privacy adds calibrated noise to computations to provide formal privacy guarantees.
Adversarial robustness studies the vulnerability of models to small, carefully crafted perturbations in inputs.
Explainable AI aims to make model decisions interpretable and transparent to humans.
Feature importance measures quantify the contribution of each input feature to the model's prediction.
SHAP values provide a unified framework for explaining individual predictions based on game theory.
LIME generates locally faithful explanations by approximating the model with an interpretable one near a specific instance.
Attention visualization highlights which parts of the input the model focuses on when making predictions.
Model compression reduces the size and computational requirements of neural networks for deployment on edge devices.
Pruning removes unnecessary weights or neurons from a network, reducing its size without significantly affecting performance.
Neural architecture search automates the design of neural network architectures using search algorithms.
Hyperparameter optimization finds the best configuration of model hyperparameters to maximize performance.
Cross-validation evaluates model performance by partitioning the data into multiple folds and training on different subsets.
Ensemble methods combine predictions from multiple models to improve overall accuracy and robustness.
Bagging trains multiple models on different bootstrap samples and averages their predictions.
Boosting sequentially trains models that focus on the mistakes of previous models, reducing bias.
Stacking trains a meta-learner to combine the predictions of multiple base models.
Random forests construct multiple decision trees and aggregate their predictions through voting.
Gradient boosting machines build trees sequentially, each correcting the residual errors of the previous ensemble.
XGBoost is an optimized implementation of gradient boosting that includes regularization and parallel processing.
LightGBM uses histogram-based splitting and leaf-wise tree growth for faster training on large datasets.
CatBoost handles categorical features natively and uses ordered boosting to reduce overfitting.
Support vector machines find the optimal hyperplane that separates classes with the maximum margin.
Kernel methods implicitly map data to higher-dimensional spaces where linear separation becomes possible.
K-nearest neighbors classifies instances based on the majority label among their k closest neighbors.
Naive Bayes applies Bayes' theorem with the assumption of feature independence, effective for text classification.
Decision trees recursively partition the feature space based on feature values, creating a tree-like structure.
Logistic regression models the probability of a binary outcome using the logistic function.
Linear regression fits a linear relationship between input features and a continuous target variable.
Ridge regression adds an L2 penalty to the loss function, shrinking coefficients toward zero.
Lasso regression adds an L1 penalty, encouraging sparsity by driving some coefficients exactly to zero.
Elastic net combines L1 and L2 penalties, balancing sparsity with coefficient shrinkage.
Principal component analysis finds orthogonal directions of maximum variance in the data.
Independent component analysis separates a multivariate signal into additive independent components.
Factor analysis models observed variables as linear combinations of latent factors plus noise.
Canonical correlation analysis finds linear combinations of two sets of variables that are maximally correlated.
Multidimensional scaling represents pairwise distances between points in a low-dimensional space.
Isomap extends MDS by preserving geodesic distances along a neighborhood graph.
Locally linear embedding reconstructs each point as a linear combination of its neighbors, preserving local geometry.
Spectral clustering uses the eigenvalues of a similarity matrix to perform dimensionality reduction before clustering.
K-means partitions data into k clusters by iteratively assigning points to the nearest centroid and updating centroids.
DBSCAN groups together points that are closely packed and marks points in low-density regions as outliers.
Gaussian mixture models represent the data distribution as a weighted sum of Gaussian components.
Expectation-maximization alternates between estimating latent variables and maximizing the likelihood of observed data.
Hidden Markov models model sequential data with hidden states that emit observable outputs.
Conditional random fields model the conditional probability of label sequences given observed features.
Graph neural networks operate on graph-structured data, aggregating information from neighboring nodes.
Graph attention networks learn attention weights for neighboring nodes, weighting their contributions dynamically.
Message passing neural networks propagate information along edges to learn node and graph representations.
Knowledge graphs represent entities and their relationships as triples, enabling structured reasoning.
Graph embeddings map nodes or entire graphs to continuous vector spaces for downstream tasks.
Temporal graph networks extend GNNs to dynamic graphs where edges and nodes change over time.
Point clouds represent 3D shapes as sets of points, processed by architectures like PointNet.
Mesh networks represent 3D surfaces as collections of vertices, edges, and faces.
Neural radiance fields synthesize novel views of a scene by mapping coordinates to color and density.
Implicit neural representations encode signals as continuous functions parameterized by neural networks.
Differentiable rendering enables gradient-based optimization of 3D scenes from 2D images.
Physics-informed neural networks incorporate physical laws as constraints during training.
Neural ordinary differential equations model continuous-time dynamics using neural networks as the derivative function.
State space models provide a framework for modeling sequential data with latent state transitions.
Mamba is a selective state space model that achieves linear complexity in sequence length.
Recurrent models process sequences step by step, while convolutional models process them in parallel with limited receptive fields.
Transformer models process all positions simultaneously using attention, enabling global context at quadratic cost.
Linear attention approximates standard attention with linear complexity by decomposing the attention matrix.
Flash attention optimizes the standard attention computation by reducing memory accesses through tiling.
Ring attention distributes the attention computation across multiple devices along the sequence dimension.
Grouped query attention shares key and value heads across multiple query heads to reduce memory during inference.
Multi-query attention uses a single key-value head shared across all query heads for faster inference.
Sliding window attention restricts each token to attend only to a local window of preceding tokens.
Sparse attention patterns reduce the quadratic cost by computing attention only for selected token pairs.
Mixture of experts routes each input token to a subset of specialized expert networks, increasing capacity without proportional cost.
Routing algorithms in MoE determine which experts process each token based on learned gating functions.
Load balancing ensures that experts receive roughly equal numbers of tokens during training.
Expert parallelism distributes different experts across devices, enabling larger total model capacity.
Token dropping discards tokens that are routed to overloaded experts, trading quality for throughput.
Auxiliary losses encourage balanced expert utilization by penalizing uneven routing distributions.
Top-k routing selects the k highest-scoring experts for each token, while top-1 routing selects only the best expert.
Soft routing weights the outputs of all experts by their gating scores, avoiding hard discretization.
Stochastic routing introduces randomness during training to improve exploration of expert assignments.
Expert capacity limits the maximum number of tokens each expert can process, preventing bottlenecks.
""" * 20


def load_trained_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        return None
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            enc = pickle.load(f)
        return enc
    except Exception:
        return None


def tokenize_with_trained(enc, text):
    return enc.encode_ordinary(text)


def hash_token(word):
    h = hashlib.sha256(word.encode("utf-8")).digest()
    return struct.unpack("<I", h[:4])[0] % VOCAB_SIZE


def tokenize_hash(text):
    tokens = []
    buf = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            buf.append(ch)
        else:
            if buf:
                tokens.append(hash_token("".join(buf)))
                buf = []
            stripped = ch.strip()
            if stripped:
                tokens.append(hash_token(stripped))
    if buf:
        tokens.append(hash_token("".join(buf)))
    return tokens


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enc = load_trained_tokenizer()
    if enc is not None:
        print(f"Using trained tokenizer from {TOKENIZER_PATH}")
        text = SAMPLE_TEXT
        tokens = tokenize_with_trained(enc, text)
        vocab_size = enc.n_vocab
    else:
        print("No trained tokenizer found, using hash-based fallback")
        text = SAMPLE_TEXT
        tokens = tokenize_hash(text)
        vocab_size = VOCAB_SIZE

    tokens_arr = np.array(tokens, dtype=np.int32)
    num_tokens = len(tokens_arr)

    with open(TOKENS_BIN, "wb") as f:
        f.write(tokens_arr.tobytes())

    meta = {"numTokens": num_tokens, "vocabSize": vocab_size}
    with open(META_JSON, "w") as f:
        json.dump(meta, f)

    print(f"Exported {num_tokens} tokens (vocab={vocab_size})")
    print(f"  {TOKENS_BIN} ({os.path.getsize(TOKENS_BIN)} bytes)")
    print(f"  {META_JSON}")


if __name__ == "__main__":
    main()
