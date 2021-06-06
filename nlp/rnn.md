[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

RNN is another paradigm of neural network where we have difference layers of cells, and each cell not only takes the cell from the previous layer as input, but also the previous cell within the same layer. This gives RNN the power to model sequence.

![RNN](../.gitbook/assets/rnn.jpeg)

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:

![An unrolled recurrent neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

![rnn1](../.gitbook/assets/rnn1.png)

- $$h_{t} = f(h_{t-1}, x_{t}; \theta)$$, where the current hidden state $$h_{t}$$ is a function $$f$$ of the previous hidden state and $$h_{t - 1}$$ the current input $$x_{t}$$. The are $$\theta$$ the parameters of the function $$f$$.

### Summary

- RNN for text, speech and time series data
- Hidden state $$h_t$$ aggregates information in the inputs $$x_0,...,x_t$$.
- RNNs can forget early inputs.
  - It forgets what it has seen eraly on
  - if it is large, $$h_t$$ is almost irrelvent to $$x_0$$.

### Number of parameters

- SimpleRNN has a parameter matrix (and perhaps an intercept vector).
- Shape of the parameter matrix is
  - $$ shape(h) \times [shape(h)+shape(x)]$$
- Only one such parameter matrix, no matter how long the sequence is.
