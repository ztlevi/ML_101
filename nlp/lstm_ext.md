# LSTM Ext.

## Standard RNN

![](<../.gitbook/assets/rnn1 (1).png>)

## Stacked RNN & LSTM

![](<../.gitbook/assets/stacked\_rnn\_1 (1).png>)

### Sample code for stacked LSTM

```python
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense

vocabulary = 10000
embedding_dim = 32
word_num = 500
state_dim = 32

model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(LSTM(state_dim, return_sequences=True, dropout=0.2))
model.add(LSTM(state_dim, return_sequences=True, dropout=0.2))
model.add(LSTM(state_dim, return_sequences=False, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```

| Layer (type)             | Output Shape    | Param  |
| ------------------------ | --------------- | ------ |
| embedding\_1 (Embedding) | (None, 500, 32) | 320000 |
| lstm\_1(LSTM)            | (None, 500, 32) | 8320   |
| lstm\_2(LSTM)            | (None, 500, 32) | 8320   |
| lstm\_3(LSTM)            | (None, 32)      | 8320   |
| dense\_1 (Dense)         | (None, 1)       | 33     |

Total params: 344,993

Trainable params: 344,993

Non-trainable params: 0

## Bidirectional RNN && LSTM

![](<../.gitbook/assets/bidirectional-rnn-1 (1).png>)

### Sample code for Bi-LSTM

```python
from keras.model import Sequential
from keras.layers import LSTM, Embedding, Dense, Bidirectional

vocabulary = 10000
embedding_dim = 32
word_num = 500
state_dim = 32

model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
```

| Layer (type)                   | Output Shape    | Param  |
| ------------------------------ | --------------- | ------ |
| embedding\_1 (Embedding)       | (None, 500, 32) | 320000 |
| bidirectional\_1 (Bidirection) | (None, 64)      | 16640  |
| dense\_1 (Dense)               | (None, 1)       | 65     |

Total params: 336,705 Trainable params: 336,705 Non-trainable params: 0

## Summary

* SimpleRNN and LSTM are two kinds of RNNs; always use LSTM instead of SimpleRNN.
* Use Bi-RNN instead of RNN whenever possible.
* Stacked RNN may be better than a single RNN layer (if n is big).
* Pretrain the embedding layer (if n is small).
