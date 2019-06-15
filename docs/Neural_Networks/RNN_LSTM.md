<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->

**Table of Contents**

- [RNN and LSTM](#rnn-and-lstm)
  - [RNN](#rnn)
  - [LSTM](#lstm)

<!-- markdown-toc end -->

# RNN and LSTM

## RNN

https://colah.github.io/posts/2015-08-Understanding-LSTMs/

RNN is another paradigm of neural network where we have difference layers of cells, and each cell not only takes the cell from the previous layer as input, but also the previous cell within the same layer. This gives RNN the power to model sequence.

![RNN](../../assets/rnn.jpeg)

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:

![An unrolled recurrent neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

- $h_{t} = f(h_{t-1}, x_{t}; \theta)$, where the current hidden state $h_{t}$ is a function $f$ of the previous hidden state and $h_{t - 1}$ the current input $x_{t}$. The are $\theta$ the parameters of the function $f$.

## LSTM

This seems great, but in practice RNN barely works due to **exploding/vanishing gradient**, which is cause by a series of multiplication of the same matrix. On the other side, it also have the problem of **long-term dependencies**. To solve this, we can use a variation of RNN, called long short-term memory (LSTM), which is capable of learning long-term dependencies.

**sigmoid** - gate function [0, 1], **tanh** - regular information to [-1, 1]

The math behind LSTM can be pretty complicated, but intuitively LSTM introduce

- input gate
- output gate
- forget gate
- memory cell (internal state)

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged.

LSTM resembles human memory: it forgets old stuff (old internal state _ forget gate) and learns from new input (input node _ input gate)

![lstm](../../assets/lstm.png)

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![A LSTM neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

1. At first, we apply **Forget gate**: $$f_{t} = \sigma(W_f \cdot [h_{t-1}, x_{t}] + b_{f})$$, caculate what information we should forget for previous information

2. The next step is to decide what new information we’re going to store in the cell state.

   - **Input gate**: $$i_{t} = \sigma(W_i \cdot [h_{t-1}, x_{t}] + b_{i})$$, a sigmoid layer decides which values we’ll update.
   - A tanh layer creates a vector of new candidate values: $$\tilde{ C_{t} } = tanh(W_{c} \cdot [h_{t-1}, x_{t}] + b_{c})$$, that could be added to the state.

3. Then we update **Memory cell C**: $$C_{t} = f_{t} * C_{t - 1} + i_{t} * \tilde{ C_{t} }$$,

   We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it∗C̃ t. This is the new candidate values, scaled by how much we decided to update each state value.

4. This output will be based on our cell state, but will be a **filtered version**.

   - First, we run a **Output gate**: $$o_{t} = \sigma(W_o \cdot [h_{t-1}, x_{t}] + b_{o})$$, which decides what parts of the cell state we’re going to output, .
   - Then, we put the cell state through tanhtanh (to push the values to be between $$[-1, 1]$$ ) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to: $$h_{t} = tanh(C_{t}) * o_{t}$$
