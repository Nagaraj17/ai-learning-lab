# AI Learning Lab - Reference Material Map

While we have several comprehensive textbooks in our `resources/references` directory, we don't need to read them cover-to-cover. Instead, we use the **"Best Tool for the Job"** approach. For each topic in our curriculum, we refer to the specific book that explains it best.

Here is the mapping of our curriculum topics to their optimal reference book:

### 1. Mathematical Foundations & Neural Networks
**Best Reference:** `Deep Learning md` (by Ian Goodfellow, Yoshua Bengio, and Aaron Courville)
*   **Why:** This is the absolute gold standard for the rigorous mathematics behind deep learning. It provides formal definitions, calculus, and linear algebra that form the bedrock of AI.
*   **Topics to use this for:**
    *   `04 - MATH - Vectors Matrices and Shapes.md`
    *   `05 - MATH - Matrix Multiplication.md`
    *   `06` to `13` - Neural Network topics (Weights, Biases, Hidden Layers, Backpropagation, Gradients, Cross-Entropy Loss)
    *   `16 - MATH - Cosine Similarity.md`
    *   `18 - MATH - Dimensionality Reduction and PCA.md`

### 2. Large Language Model Architecture & Intuition
**Best Reference:** `LLMs-resource.md` (*Hands-On Large Language Models* by Jay Alammar & Maarten Grootendorst)
*   **Why:** Jay Alammar is famous for his highly visual and intuitive explanations of complex architectures like the Transformer. This book is the best for understanding the *mechanics* and *intuition* of LLMs without getting bogged down in production engineering.
*   **Topics to use this for:**
    *   `02 - LM - Tokens and Vocabulary.md`
    *   `03 - LM - One-Hot Encoding.md`
    *   `14 - LM - Embeddings.md`
    *   `17 - LM - Tokenizers.md`
    *   `Weekly Curriculum/03-Attention`
    *   `Weekly Curriculum/04-Transformer-Block`

### 3. AI Applications, RAG, and Production Engineering
**Best Reference:** `AI Engineering by Chip Huyen.md` (*AI Engineering: Building Applications with Foundation Models* by Chip Huyen)
*   **Why:** This book bridges the gap between training a model and actually building a useful product with it. It focuses on the engineering challenges of taking foundation models into production.
*   **Topics to use this for:**
    *   **Prompt Engineering & Security** (Chapter 5)
    *   **RAG (Retrieval-Augmented Generation) & Agents** (Chapter 6)
    *   **Finetuning** (Chapter 7)
    *   **Evaluation & Benchmarking** (Chapters 3 & 4)
    *   **Inference Optimization (Latency/Cost)** (Chapter 9)
    *   *Note: These topics represent the next phase of our journey after we complete the Tiny-GPT weekly curriculum.*

---
**Protocol:** When creating or updating a topic note, we will consult the **Best Reference** listed above to extract the intuition, definitions, and real-world examples, ensuring we only pull the most relevant, high-quality explanations for our beginner-first curriculum.
