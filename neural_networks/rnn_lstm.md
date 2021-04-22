# RNN & LSTM

## RNN

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

RNN is another paradigm of neural network where we have difference layers of cells, and each cell not only takes the cell from the previous layer as input, but also the previous cell within the same layer. This gives RNN the power to model sequence.

![RNN](../.gitbook/assets/rnn.jpeg)

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:

![An unrolled recurrent neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

* $$h_{t} = f(h_{t-1}, x_{t}; \theta)$$, where the current hidden state $$h_{t}$$ is a function $$f$$ of the previous hidden state and $$h_{t - 1}$$ the current input $$x_{t}$$. The are $$\theta$$ the parameters of the function $$f$$.

## LSTM

This seems great, but in practice RNN barely works due to **exploding/vanishing gradient**, which is cause by a series of multiplication of the same matrix. On the other side, it also have the problem of **long-term dependencies**. To solve this, we can use a variation of RNN, called long short-term memory \(LSTM\), which is capable of learning long-term dependencies.

**sigmoid** - gate function \[0, 1\], **tanh** - regular information to \[-1, 1\]

![](../.gitbook/assets/LSTM3-gate.png)

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

The math behind LSTM can be pretty complicated, but intuitively LSTM introduce

* input gate
* output gate
* forget gate
* memory cell \(internal state\)

![](../.gitbook/assets/LSTM3-C-line.png)

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged.

LSTM resembles human memory: it forgets old stuff \(old internal state  _forget gate\) and learns from new input \(input node_  input gate\)

![lstm](../.gitbook/assets/lstm.png)

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![A LSTM neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

### Step-by-Step LSTM Walk Through

![](../.gitbook/assets/LSTM3-focus-f.png)

1. At first, we apply **Forget gate**: $$f_{t} = \sigma(W_f \cdot [h_{t-1}, x_{t}] + b_{f})$$, caculate what information we should forget for previous information

   ![](../.gitbook/assets/LSTM3-focus-i.png)

2. The next step is to decide what new information we’re going to store in the cell state.

   * **Input gate**: $$i_{t} = \sigma(W_i \cdot [h_{t-1}, x_{t}] + b_{i})$$, a sigmoid layer decides which values we’ll update.
   * A tanh layer creates a vector of new candidate values: $$\tilde{ C_{t} } = tanh(W_{c} \cdot [h_{t-1}, x_{t}] + b_{c})$$, that could be added to the state.

   ![](../.gitbook/assets/LSTM3-focus-C.png)

3. Then we update **Memory cell C**: $$C_{t} = f_{t} * C_{t - 1} + i_{t} * \tilde{ C_{t} }$$,

   We multiply the old state by $$f_t$$, forgetting the things we decided to forget earlier. Then we add $$i_t * \tilde{C_t}$$. This is the new candidate values, scaled by how much we decided to update each state value.

   ![](../.gitbook/assets/LSTM3-focus-o.png)

4. This output will be based on our cell state, but will be a **filtered version**.
   * First, we run a **Output gate**: $$o_{t} = \sigma(W_o \cdot [h_{t-1}, x_{t}] + b_{o})$$, which decides what parts of the cell state we’re going to output, .
   * Then, we put the cell state through tanhtanh \(to push the values to be between $$[-1, 1]$$ \) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to: $$h_{t} = tanh(C_{t}) * o_{t}$$

#### Why solve vanishing gradient?

Details from [here](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)

* The additive update function for the cell state gives a derivative thats much more ‘well behaved’
* The **gating functions allow the network to decide how much the gradient vanishes**, and can take on different values at each time step. The values that they take on are learned functions of the current input and hidden state.

### \(Optional\) Implementation

FIXME Remove redundant codes

* Part of the codes demonstrating LSTM

  > **Note**: activation='tanh', recurrent\_activation='hard\_sigmoid'

  ```python
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name="kernel",
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
    )
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name="recurrent_kernel",
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
    )

    self.kernel_i = self.kernel[:, : self.units]
    self.kernel_f = self.kernel[:, self.units : self.units * 2]
    self.kernel_c = self.kernel[:, self.units * 2 : self.units * 3]
    self.kernel_o = self.kernel[:, self.units * 3 :]

    self.recurrent_kernel_i = self.recurrent_kernel[:, : self.units]
    self.recurrent_kernel_f = self.recurrent_kernel[:, self.units : self.units * 2]
    self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2 : self.units * 3]
    self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3 :]

    x_i = K.dot(inputs_i, self.kernel_i)
    x_f = K.dot(inputs_f, self.kernel_f)
    x_c = K.dot(inputs_c, self.kernel_c)
    x_o = K.dot(inputs_o, self.kernel_o)

    i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
    f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
    o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))
  ```

* Take an example here:

  ```python
  # LSTM for sequence classification in the IMDB dataset
  import numpy

  from keras.datasets import imdb
  from keras.layers import LSTM, Dense
  from keras.layers.embeddings import Embedding
  from keras.models import Sequential
  from keras.preprocessing import sequence

  # fix random seed for reproducibility
  numpy.random.seed(7)
  # load the dataset but only keep the top n words, zero the rest
  top_words = 5000
  (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
  # truncate and pad input sequences
  max_review_length = 500
  X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
  X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
  # create the model
  embedding_vecor_length = 32
  model = Sequential()
  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(LSTM(100))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
  print(model.summary())
  model.fit(X_train, y_train, epochs=3, batch_size=64)
  # Final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1] * 100))
  ```

  * Shapes
    * The output shape of the Embedding layer is \(?, 500, 32\).
    * $$C_t$$: \(?, 100\)
    * $$h_t$$: \(?, 100\)

  The calculation for forget gate $$f_{t} = \sigma(W_f \cdot [h_{t-1}, x_{t}] + b_{f})$$ is composed of:

  $$
  \sigma ( (?,32) \cdot (32,100) + (?, 100) \cdot (100, 100) + (1, 100))
  $$

