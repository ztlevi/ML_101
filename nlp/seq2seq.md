# Seq2Seq

## 1. Tokenization & Build Dictionary

* input\_texts =&gt; \[Eng\_Tokenizer\] =&gt; input\_tokens
* target\_texts =&gt; \[Deu\_Tokenizer\] =&gt; target\_tokens
  * Use 2 different tokenizers for the 2 languages.
  * Then build 2 different dictionaries.
* Tokenization can be char-level or word-level

```text
Eng_Tokenizer: "I_am_okay." => ['i', '_', 'a', 'm', ..., 'a', 'y']
Deu_Tokenizer: "Es geht mir gut" => ['e', 's', '_', ..., 'u', 't']
```

### Question: Why 2 different tokenizers and dictionaries?

Answer: In the char-level, languages have different alphabets/chars.

* English: A a, B b, C c …, Z z. \(26 letters × 2\).
* German: 26 letters, 3 umlauts \(Ä,Ö,Ü\), and one ligature \(ß\).
* Greek: Α α, Β β, Γ γ, Δ δ, …, Ω ω. \(24 letters × 2\).
* Chinese: 金 木 水 火 土 … 赵 钱 孙 李 \(a few thousands characters\).

### Question: Why 2 different tokenizers and dictionaries?

Answer: In the word-level, languages have different vocabulary.

## 2. One-Hot Encoding

![](../.gitbook/assets/rnn-5.png)

## 3. Training Seq2Seq Model

![](../.gitbook/assets/rnn-6.png)

![](../.gitbook/assets/rnn-7.png)

## 4. Inference

![](../.gitbook/assets/rnn-8.png)

![](../.gitbook/assets/rnn-9.png)

![](../.gitbook/assets/rnn-10.png)

![](../.gitbook/assets/rnn-11.png)

## Summary

![](../.gitbook/assets/rnn-12.png)

* Encoder’s final states \(𝐡 &lt; and 𝐜 &lt; \) have all the information of the English sentence.
* If the sentence is long, the final states have forgotten early inputs.
* Bi-LSTM \(left-to-right and right-to-left\) has longer memory.
* Use Bi-LSTM in the encoder; use unidirectional LSTM in the decoder.
* Word-level tokenization instead of char-level.
  * The average length of English words is 4.5 letters.
  * The sequences will be 4.5x shorter.
  * Shorter sequence -&gt; less likely to forget.
* But you will need a large dataset!
  * **of \(frequently used\) chars is ~**$$10^2$$ **\) -&gt; one-hot suffices.**
  * **of \(frequently used\) words is ~**$$10^4$$ **-&gt; must use embedding.**
  * Embedding Layer has many parameters -&gt; overfitting!

