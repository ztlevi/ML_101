# BLEU

## Precision and modified precision

Example of poor machine translation output with high precision

| Candidate   | the   | the | the | the | the | the | the |
| ----------- | ----- | --- | --- | --- | --- | --- | --- |
| Reference 1 | the   | cat | is  | on  | the | mat |     |
| Reference 2 | there | is  | a   | cat | on  | the | mat |

Of the seven words in the candidate translation, all of them appear in the reference translations. Thus the candidate text is given a unigram precision of,

$$
P=\frac{m}{w_t}=\frac{7}{7}=1
$$

where![\~m](https://wikimedia.org/api/rest\_v1/media/math/render/svg/a8acad9f788979dc8ab742808922342fa9638f74) is number of words from the candidate that are found in the reference, and![\~w\_{t}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/b1851ecca3f4ed24d37ca301b60fd03e3b1de896) is the total number of words in the candidate. This is a perfect score, despite the fact that the candidate translation above retains little of the content of either of the references.

The modification that BLEU makes is fairly straightforward. For each word in the candidate translation, the algorithm takes its maximum total count, ![\~m\_{max}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/073e0b6734439b09824af280cbfc04311307e957), in any of the reference translations. In the example above, the word "the" appears twice in reference 1, and once in reference 2. Thus![\~m\_{max} = 2](https://wikimedia.org/api/rest\_v1/media/math/render/svg/cdbff6244c23782c04f19b4ea6cd9e2cc4f8c1fa).

For the candidate translation, the![m\_{w}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/1bd173f48c4afc862b28f62d46b4ff220e64f016) of each word is clipped to a maximum of![m\_{max}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/1bbaf427c6254840201f7d9e20d21e29bc635682) for that word. In this case, "the" has![\~m\_{w} = 7](https://wikimedia.org/api/rest\_v1/media/math/render/svg/43afd971be144e2ed0200837d10a1d66df01d55d) and ![\~m\_{max}=2](https://wikimedia.org/api/rest\_v1/media/math/render/svg/cdbff6244c23782c04f19b4ea6cd9e2cc4f8c1fa), thus![\~m\_{w}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/b49382562e4afb4fa88a52784e73e1cb363c5a75) is clipped to 2. These clipped counts![\~m\_{w}](https://wikimedia.org/api/rest\_v1/media/math/render/svg/b49382562e4afb4fa88a52784e73e1cb363c5a75) are then summed over all distinct words in the candidate. This sum is then divided by the total number of unigrams in the candidate translation. In the above example, the modified unigram precision score would be:

$$
P=\frac{2}{7}
$$

## bigram modified precision

In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision metric using [n-grams](https://en.wikipedia.org/wiki/N-gram).

Here is an example of 2-gram (bigram) modified prevision calculation:

|                        |       |     |     |     |     |     |     |
| ---------------------- | ----- | --- | --- | --- | --- | --- | --- |
| Candidate ($$\hat y$$) | the   | cat | the | cat | on  | the | mat |
| Reference 1            | the   | cat | is  | on  | the | mat |     |
| Reference 2            | there | is  | a   | cat | on  | the | mat |

|         |           |                  |
| ------- | --------- | ---------------- |
| bigram  | $$Count$$ | $$Count_{clip}$$ |
| the cat | 2         | 1                |
| cat the | 1         | 0                |
| cat on  | 1         | 1                |
| on the  | 1         | 1                |
| the mat | 1         | 1                |

$$
Modified\_Precision=\frac{1+0+1+1+1}{2+1+1+1+1}=\frac{4}{6}
$$

Where $$Count_{clip}$$means the max count of the bigrams appears in the references.

## N-gram modified precision

$$
P_n=\frac{\sum_{n-gram \in \hat y} Count_{clip}(n-gram) }{\sum_{n-gram \in \hat y} Count(n-gram)}
$$

$$p_n$$=Bleu score on n-grams only

### Combined blue score

The length which has the "highest correlation with monolingual human judgements"[\[1\]](https://en.wikipedia.org/wiki/BLEU#endnote\_Papineni2002c) was found to be four. The unigram scores are found to account for the adequacy of the translation, how much information is retained. The longer n-gram scores account for the fluency of the translation, or to what extent it reads like "good English".

$$
Bleu = BP \times exp(\frac{1}{4} \sum_{n=1}^4p_n )
$$

Where BP is brevity penlty. Machine tends to generate shorter translation because the score will be higher. Add the penality so that it would generate longer sentences.

$$
BP=\begin{cases} 1 \text{ if candidate_output_length > reference_output_length}\\ exp(1-\text{candidate_output_length/reference_output_length}) \text{ otherwise} \end{cases}
$$

## Calculate BLEU score

The Python Natural Language Toolkit Library (NLTK) provides an implementation of BLEU scoring. You can use it to evaluate the generated text by comparing it with the reference text.

### Statement BLEU score

NLTK provides the [sentence\_bleu()](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu\_score.sentence\_bleu) function to evaluate candidate sentences based on one or more reference sentences .

The reference sentence must be provided as a list of sentences, where each sentence is a list of tokens. Candidate sentences are provided as a list of tokens. E.g:

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)
```

Running this example will output a full score, because the candidate sentence exactly matches one of the reference sentences.

```
1.0
```

### Corpus BLEU score

NLTK also provides a function called [corpus\_bleu()](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu\_score.corpus\_bleu) to calculate multiple sentences (such as paragraphs or Document) BLEU score.

The reference text must be specified as a list of documents, where each document is a list of reference sentences, and each replaceable reference sentence is also a list of tokens, which means that the document list is a list of lists of token lists. Candidate documents must be specified as a list, where each file is a list of tokens, which means that the candidate document is a list of token lists.

This sounds a bit confusing; the following are examples of two reference documents for one document.

```python
# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score)
```

Run this example and output full marks as before.

```
1.0
```

### Cumulative and individual BLEU scores

The BLEU scoring method provided in NLTK allows you to assign weights to different n-tuples when calculating BLEU scores.

This gives you the flexibility to calculate different types of BLEU scores, such as individual and cumulative n-gram scores.

Let's take a look.

### Individual N-Gram score

An individual N-gram score is a score for matching n-tuples in a specific order, such as a single word (called 1-gram) or word pair (called 2-gram or bigram).

The weights are specified as an array, where each index corresponds to an n-tuple in the corresponding order. To calculate the BLEU score for 1-gram matching, you can specify the 1-gram weight as 1, and for 2, 3 and 4 gram, the weight is 0, that is, the weight is (1,0,0,0). E.g:

```python
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this','is','small','test']]
candidate = ['this','is','a','test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)
```

Running this example will output a score of 0.5.

```
0.75
```

We can repeat this example, and run the sentence for each n-gram from 1 yuan to 4 yuan as follows:

```python
# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this','is','a','test']]
candidate = ['this','is','a','test']
print('Individual 1-gram: %f'% sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f'% sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f'% sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f'% sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
```

Running the example, the results are as follows:

```
Individual 1-gram: 1.000000
Individual 2-gram: 1.000000
Individual 3-gram: 1.000000
Individual 4-gram: 1.000000
```

Although we can calculate a separate BLEU score, this is not the original intention of using this method, and the resulting score does not have much meaning or seems to be illustrative.

### Cumulative N-Gram score

The cumulative score refers to the calculation of all individual n-gram scores from 1 to n, and they are weighted by calculating the weighted geometric mean.

By default, `sentence_bleu()` and `corpus_bleu()` scores calculate the accumulated 4-tuple BLEU scores, also known as BLEU-4 scores.

BLEU-4 weights the numbers of 1-tuples, 2-tuples, 3-tuples and 4-tuples as 1/4 (25%) or 0.25. E.g:

```python
# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this','is','small','test']]
candidate = ['this','is','a','test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)
```

Run this example and output the following scores:

```
0.707106781187
```

The cumulative and single 1-tuple BLEU use the same weight, which is (1,0,0,0). The BLEU score of the accumulated 2-tuple is calculated as a 1-tuple and a 2-tuple with a weight of 50% respectively, and the accumulated 3-tuple BLEU is calculated as a 1-tuple, and a 2-tuple and a 3-tuple are respectively assigned a weight of 33%.

Let us specify by calculating the cumulative scores of BLEU-1, BLEU-2, BLEU-3 and BLEU-4:

```python
# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['this','is','small','test']]
candidate = ['this','is','a','test']
print('Cumulative 1-gram: %f'% sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f'% sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f'% sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f'% sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
```

Running this example outputs the following scores. The results vary greatly, and are more expressive than individual n-gram scores.

```
Cumulative 1-gram: 0.750000
Cumulative 2-gram: 0.500000
Cumulative 3-gram: 0.632878
Cumulative 4-gram: 0.707107
```

When describing the performance of the text generation system, the cumulative score from BLEU-1 to BLEU-4 is usually reported.

### Run the example

In this section, we try to use some examples to further gain intuition for BLEU scoring.

At the sentence level, we use the following reference sentence to illustrate:

> the quick brown fox jumped over the lazy dog

First, let's look at a perfect score.

```python
# prefect match
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['the','quick','brown','fox','jumped','over','the','lazy','dog']
score = sentence_bleu(reference, candidate)
print(score)
```

Running the example outputs a perfect match score.

```
1.0
```

Next, let's change a word and change "_quick_" to "_fast_ ".

```python
# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['the','fast','brown','fox','jumped','over','the','lazy','dog']
score = sentence_bleu(reference, candidate)
print(score)
```

The result is a slight drop in scores.

```
 0.7506238537503395
```

Try to change two words, change "_quick_" to "_fast_" and "_lazy_" to "_sleepy_ ".

```python
# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['the','fast','brown','fox','jumped','over','the','sleepy','dog']
score = sentence_bleu(reference, candidate)
print(score)
```

Running this example, we can see that the score drops linearly.

```
0.4854917717073234
```

What if all the words in the candidate sentence are different from those in the reference sentence?

```python
# all words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['a','b','c','d','e','f','g','h','i']
score = sentence_bleu(reference, candidate)
print(score)
```

We got a worse score.

```
0.0
```

Now, let's try a candidate sentence that has fewer words than the reference sentence (for example, discarding the last two words), but these words are all correct.

```python
# shorter candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['the','quick','brown','fox','jumped','over','the']
score = sentence_bleu(reference, candidate)
print(score)
```

The result is very similar to the previous case where there were two word errors.

```
0.7514772930752859
```

What if we adjust the candidate sentence to two more words than the reference sentence?

```python
# longer candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
candidate = ['the','quick','brown','fox','jumped','over','the','lazy','dog','from','space']
score = sentence_bleu(reference, candidate)
print(score)
```

Once again, we can see that our intuition is valid

## Reference

* [https://en.wikipedia.org/wiki/BLEU](https://en.wikipedia.org/wiki/BLEU)
* [https://machinelearningmastery.com/calculate-bleu-score-for-text-python/](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
* [https://www.youtube.com/watch?v=DejHQYAGb7Q](https://www.youtube.com/watch?v=DejHQYAGb7Q)
