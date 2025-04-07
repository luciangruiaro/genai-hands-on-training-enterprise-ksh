# 📘 Understanding Embeddings in NLP

Embeddings are a cornerstone of modern NLP, enabling models to understand words and texts in a numerical, meaningful
way. This guide walks you through the **theory**, **practice**, and **code** to master embeddings, from One-Hot to
Word2Vec and beyond.

---

## 📌 Table of Contents

1. [Cosine Similarity](#cosine-similarity)
2. [Word Vectorization Methods](#word-vectorization-methods)
    - One-Hot Encoding
    - TF-IDF
    - Word Embeddings
3. [What is Gensim?](#what-is-gensim)
4. [Embeddings vs Hashing](#embeddings-vs-hashing)
5. [How Embedding Models Work](#how-embedding-models-work)
6. [Neural Network Approximation](#neural-network-approximation)
7. [Word2Vec Algorithms & Architecture](#word2vec-algorithms--architecture)
8. [Word2Vec and Modern Embeddings](#word2vec-and-modern-embeddings)
9. [Hidden Layer Representation: Why No Collisions?](#hidden-layer-representation-why-no-collisions)
10. [Matrix View of Embeddings](#matrix-view-of-embeddings)
11. [Softmax in Word2Vec](#softmax-in-word2vec)

---

## 🔹 Cosine Similarity

Cosine similarity measures the angle between two vectors, helpful for comparing word meanings.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([[1, 2, 3]])
vec2 = np.array([[2, 4, 6]])

similarity = cosine_similarity(vec1, vec2)
print("Cosine Similarity:", similarity[0][0])
```

---

## 🔸 Word Vectorization Methods

### 1. One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

words = [["king"], ["queen"], ["man"], ["woman"]]
encoder = OneHotEncoder(sparse=False)
one_hot = encoder.fit_transform(words)

print(encoder.categories_)
print(one_hot)
```

🔹 **Limitation**: High dimensionality, no semantic similarity.

---

### 2. TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["the cat sat", "the dog barked"]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
```

🔹 Captures word importance in the document but not meaning.

---

### 3. Word Embeddings (Dense Vectors)

```python
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")  # 50-dim embeddings
print(model["king"])  # Vector for 'king'

similarity = model.similarity("king", "queen")
print("Similarity:", similarity)
```

🔹 Trained on context to capture semantic meaning.

---

## 🧰 What is Gensim?

**Gensim** is a Python library for unsupervised topic modeling and document similarity analysis. It supports models like
Word2Vec, FastText, LDA, and more.

Install:

```bash
pip install gensim
```

Gensim abstracts complex training and vector operations behind simple APIs.

---

## ⚖️ Embeddings vs Hashing

| Feature   | Embeddings            | Hashing                    |
|-----------|-----------------------|----------------------------|
| Type      | Learned dense vectors | Rule-based index           |
| Collision | Avoided via training  | Possible (hash collisions) |
| Meaning   | Captures semantics    | No meaning awareness       |
| Dimension | Lower + dense         | Sparse / depends on hash   |

📌 *Hashing* is fast, but **not semantic**. Embeddings learn relationships.

---

## ⚙️ How Embedding Models Work

An embedding layer is a matrix `V x D`:

- `V`: Vocabulary size
- `D`: Dimension of vector

Each word is mapped to a row. Training updates these vectors based on context using **backpropagation**.

---

## 🧠 Neural Network Approximation

You can approximate word embeddings using a **shallow neural network**.

```text
Input word (One-hot) --> Hidden Layer (Linear Projection) --> Softmax output (Context prediction)
```

No activation on hidden layer – just a dot product. Training is done using:

- **Skip-gram** or **CBOW**
- **Negative Sampling** or **Hierarchical Softmax**

---

## 🔍 Word2Vec Algorithms & Architecture

**Word2Vec** has two main modes:

- **CBOW (Continuous Bag of Words)**: Predicts a word from surrounding context.
- **Skip-Gram**: Predicts surrounding words from a target word.

📐 Internally:

- Input layer: One-hot word vector
- Hidden layer: Linear transformation
- Output: Softmax or sampled context words

---

## 🧠 Is Word2Vec a Neural Network?

Yes, it's a **2-layer shallow neural net**:

- No activation function in hidden layer
- Linear projection learns the embeddings

```python
from gensim.models import Word2Vec

sentences = [["king", "queen", "man", "woman"]]
model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1)
print(model.wv["king"])
```

---

## 🔍 Are OpenAI and Others Using Word2Vec?

No. Modern embeddings use **transformer-based architectures** (e.g. BERT, GPT) rather than Word2Vec.

However, Word2Vec laid the groundwork for:

- Dense representations
- Transfer learning
- Token embeddings (before fine-tuning)

---

## 💡 Hidden Layer Representation: Why No Collisions?

Even with millions of words, each word gets a **unique row in the weight matrix**, updated independently during
training.

There are no collisions because:

- Each word is indexed by its unique ID
- Embeddings are learned, not computed via hash

---

## 🧮 Matrix View of Embeddings

Imagine an embedding matrix `E`:

```
E = [
  [0.1, 0.2, 0.3],  # "king"
  [0.4, 0.5, 0.6],  # "queen"
  ...
]
```

The embedding for word `i` is simply `E[i]`.

Training updates only the row(s) corresponding to current input/context.

---

## ❓ Why No Activation Function?

- **Hidden Layer** = purely linear: `output = W * input`
- No activation function is needed because we're **learning linear projections**
- Activations would distort the geometry of the learned space

Backpropagation still happens: gradients flow into the embedding matrix to update vectors.

---

## 🎯 Why Softmax at Output?

Used only for training:

- Softmax converts scores to probabilities of context words
- The objective is to **maximize the probability of true context words**

At inference, we don’t use softmax—we only extract the hidden layer as the embedding vector.

---

## ✅ Summary

| Concept        | Insight                                                |
|----------------|--------------------------------------------------------|
| One-Hot        | Sparse, no meaning                                     |
| TF-IDF         | Importance, not meaning                                |
| Embeddings     | Semantic, dense                                        |
| Word2Vec       | 2-layer net, predicts context                          |
| Gensim         | Python lib for topic modeling & embeddings             |
| Hashing        | Faster, no meaning, risk of collisions                 |
| Softmax        | Only used during training for classification objective |
| No activations | Ensures geometric consistency of vector space          |

---

## ✅ Suggested Further Reading

- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Gensim documentation: https://radimrehurek.com/gensim/
- Word2Vec negative
  sampling: https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html
