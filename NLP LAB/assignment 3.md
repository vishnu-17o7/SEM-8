---
header-includes:
  - \usepackage{geometry}
  - \geometry{margin=1in}
  - \usepackage{eso-pic}
  - \AddToShipoutPictureBG*{\AtPageLowerLeft{\put(0,0){\rule{\paperwidth}{2pt}}\put(0,\paperheight){\rule{\paperwidth}{2pt}}\put(0,0){\rule{2pt}{\paperheight}}\put(\paperwidth,0){\rule{2pt}{\paperheight}}}}
---

# Assignment 3

## 1. Compare HMM and CRF models for sequence labelling tasks.

### Hidden Markov Models (HMMs)
Hidden Markov Models are generative probabilistic models used for sequence labeling tasks such as part-of-speech tagging, named entity recognition, and speech recognition. HMMs model the joint probability of observed sequences and hidden states, assuming that the current state depends only on the previous state (Markov property) and that observations depend only on the current state.

Key components of HMM:
- **States**: Hidden states representing labels (e.g., POS tags).
- **Observations**: Visible sequence (e.g., words).
- **Transition probabilities**: Probability of moving from one state to another.
- **Emission probabilities**: Probability of observing a symbol given a state.
- **Initial state probabilities**: Probability of starting in each state.

HMMs use algorithms like Viterbi for decoding (finding the most likely sequence of states) and Baum-Welch for training.

### Conditional Random Fields (CRFs)
Conditional Random Fields are discriminative models that directly model the conditional probability of labels given the observations, without assuming independence between observations. CRFs are particularly effective for sequence labeling because they can capture dependencies between labels and features across the entire sequence.

Key features:
- **Conditional modeling**: P(labels | observations)
- **Feature functions**: Define relationships between labels and observations.
- **Global normalization**: Ensures probabilities sum to 1 over all possible label sequences.
- **Linear-chain CRFs**: Most common for sequence data, where labels depend on previous labels.

CRFs use maximum likelihood estimation for training and Viterbi for inference.

### Comparison
| Aspect | HMM | CRF |
|--------|-----|-----|
| **Model Type** | Generative | Discriminative |
| **Independence Assumptions** | Assumes observations are independent given states; states follow Markov chain | No independence assumptions; models dependencies globally |
| **Feature Flexibility** | Limited to emission and transition probabilities | Highly flexible; can incorporate arbitrary features and dependencies |
| **Handling Long-Range Dependencies** | Struggles due to Markov assumption | Better at capturing long-range dependencies |
| **Training** | EM algorithm (Baum-Welch) | Gradient-based optimization (e.g., L-BFGS) |
| **Performance** | Good for simple tasks; may underperform with complex features | Generally superior for sequence labeling tasks with rich features |
| **Computational Complexity** | Efficient for inference | More computationally intensive, especially for long sequences |

HMMs are simpler and faster but make strong assumptions that may not hold in real data. CRFs, while more complex, provide better accuracy by modeling conditional dependencies directly.

## 2. Explain LSTM architecture and how it solves the vanishing gradient problem.

### LSTM Architecture
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data. Unlike standard RNNs, LSTMs use a memory cell and three gates to control the flow of information:

1. **Forget Gate**: Decides what information to discard from the cell state.
2. **Input Gate**: Determines what new information to store in the cell state.
3. **Output Gate**: Controls what parts of the cell state to output.

The LSTM cell processes input at each time step through these gates, updating the cell state and hidden state.

### Solving the Vanishing Gradient Problem
Standard RNNs suffer from vanishing gradients during backpropagation through time (BPTT), where gradients diminish exponentially as they propagate backward, making it difficult to learn long-term dependencies.

LSTMs address this by:
- **Gated mechanism**: Gates control information flow, allowing gradients to flow through the cell state with minimal decay.
- **Constant error carousel**: The cell state acts as a conveyor belt, carrying information across many time steps with linear interactions.
- **Selective memory**: Forget and input gates allow the network to selectively remember or forget information, preventing gradient explosion or vanishing.

This enables LSTMs to maintain gradients over long sequences, effectively capturing dependencies that span hundreds of time steps.

## 3. Discuss the role of Transformers in language modelling and text generation.

### Transformer Architecture
Transformers, introduced in the paper "Attention is All You Need" by Vaswani et al., revolutionized NLP by relying entirely on attention mechanisms without recurrence or convolution. Key components:

- **Self-Attention**: Allows each position in the sequence to attend to all other positions.
- **Multi-Head Attention**: Multiple attention heads capture different aspects of relationships.
- **Positional Encoding**: Adds position information since transformers don't have inherent sequence order.
- **Feed-Forward Networks**: Applied to each position independently.
- **Layer Normalization and Residual Connections**: Aid in training stability.

### Role in Language Modelling
Transformers excel in language modeling by capturing complex dependencies and contextual relationships:

- **Bidirectional Models (e.g., BERT)**: Use masked language modeling to predict masked tokens based on context from both directions. BERT has been foundational for many NLP tasks.
- **Unidirectional Models (e.g., GPT series)**: Predict the next token given previous tokens, enabling generative capabilities.

Transformers can model long-range dependencies effectively due to self-attention, which computes pairwise interactions across the entire sequence in parallel.

### Role in Text Generation
Transformers have become the standard for text generation:

- **GPT Models**: Use decoder-only transformers for autoregressive generation. GPT-3 and ChatGPT demonstrate impressive capabilities in generating coherent, contextually appropriate text.
- **Fine-tuning**: Models like GPT can be fine-tuned on specific domains for controlled generation.
- **Sampling Strategies**: Techniques like top-k sampling and nucleus sampling control randomness and quality.
- **Applications**: Creative writing, code generation, dialogue systems, and more.

The parallelizable nature of transformers allows for efficient training on large datasets, leading to more powerful models.

## 4. Explain encoder–decoder models with RNNs for machine translation.

### Encoder-Decoder Architecture
Encoder-decoder models are a framework for sequence-to-sequence (seq2seq) tasks like machine translation. They consist of two main components:

1. **Encoder**: Processes the input sequence and compresses it into a fixed-length context vector.
2. **Decoder**: Generates the output sequence based on the context vector.

Both encoder and decoder are typically RNNs (often LSTMs or GRUs) that process sequences step-by-step.

### Encoder
- Takes the source language sentence as input.
- Processes each word sequentially, updating hidden states.
- The final hidden state (or a combination of states) becomes the context vector representing the entire input sentence.

### Decoder
- Starts with the context vector as initial hidden state.
- Generates output words one by one, using previous predictions as input.
- At each step, produces a probability distribution over the target vocabulary.
- Uses teacher forcing during training (ground truth as input) and greedy/beam search during inference.

### Training and Inference
- **Training**: Maximize likelihood of target sequence given source.
- **Inference**: Use beam search to find most likely translation.
- **Limitations**: Fixed context vector can lose information for long sentences; attention mechanisms were added to address this.

This architecture laid the foundation for modern neural machine translation systems.