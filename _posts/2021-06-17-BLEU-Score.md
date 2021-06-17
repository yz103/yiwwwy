---
title:  "BLEU Score"
tags: ML/DL
---

BLEU, bilingual evaluation understudy, is a standard algorithm to evaluate the quality of translated text in machine translation. For example, we want to translate a French sentence into English, as we know, there is no single right answer for translation, and different people may have different ways to translate the same sentences. So we will choose several sentences translated from human as the reference texts. BLEU provides a way to measure the similarity between the translated text from machine with the references texts.

#### Algorithm

Suppose we have a candidate translation $$c$$ and a set of $$k$$ reference translations $$r_1,\dots,r_k$$.

- Firstly compute the modified n-gram precision $$p_n$$ for the candidate text

  $$
  p_n = \frac{\sum_{ngram\in c}min(max_{i=1,\dots,k}Count_{ri}(ngram),Count_c(ngram))}{\sum_{ngram\in c}Count_c(ngram)}
  $$

- Next compute the brevity penalty $$BP$$. $$BP$$ will penalize the candidate if the length of candidate is very short comparing with the reference sentences.  Let $$len(c)$$ be the length of $$c$$ and let $$len(r)$$ be the length of the reference translation that is closest to $$len(c)$$ (in the case of two equally-close reference translation lengths, choose $$len(r)$$ as the shorter one)

  $$
  BP = 
  \begin{cases}
  1\quad&\text{ if len(c)}\ge\text{ len(r)}\\
  exp(1-\frac{len(r)}{len(c)})\quad&\text{otherwise}
  \end{cases}
  $$

- Lastly, the BLEU score for candidate $$c$$ with respect to $$r_1,\dots,r_k$$ is 

  $$
  BLEU = BP\times epx(\sum_{n=1}^{4}\lambda_n logp_n)
  $$
  
  where $$\lambda_1,\lambda_2,\lambda_3,\lambda_4$$ are weights that sum to 1. The log here is natural log. And here we consider until 4-gram.



#### Example

​	Reference Translation $$r_1$$: love can always find a way

​	Refence translation $$r_2$$: love makes anything possible

​	Candidate $$c$$: the love can always do

Suppose we only consider unigram, bigram and 3-grams. 

**Calculate $$p_1$$**

We first compute the unigram precision for candidate $$c$$.

The unigram in c is "the","love","can","always","do" and the count for each is 1. So the total number of unigram is just 5.

| Unigram in Candidate | Count  in Candidate | Count in $$r_1$$ | Count in $$r_2$$ | Max(Count in $$r_1$$, Count in $$r_2$$) |
| -------------------- | ------------------- | ---------------- | ---------------- | --------------------------------------- |
| "the"                | 1                   | 0                | 0                | 0                                       |
| "love"               | 1                   | 1                | 1                | 1                                       |
| "can"                | 1                   | 1                | 0                | 1                                       |
| "always"             | 1                   | 1                | 0                | 1                                       |
| "do"                 | 1                   | 0                | 0                | 0                                       |
| Total                | 5                   | 3                | 1                | 3                                       |

The last column keeps the maximum count between the reference texts and note that the maximum will be bounded by the respective count in candidate which means if the maximum count is greater than the respective count in candidate then we use the count in candidate. 

Therefore, from the table, $$p_1=\frac{3}{5}$$

**Calculate $$p_2$$**

| Bigram in Candidate | Count in Candidate | Count in r1 | Count in r2 | Max(c in r1, c in r2) |
| ------------------- | ------------------ | ----------- | ----------- | --------------------- |
| "the love"          | 1                  | 0           | 0           | 0                     |
| "love can"          | 1                  | 1           | 0           | 1                     |
| "can always"        | 1                  | 1           | 0           | 1                     |
| "always do"         | 1                  | 0           | 0           | 0                     |
| Total               | 4                  |             |             | 2                     |

Similarly from the table $$p_2 = \frac{1}{2}$$

**Calculate $$p_3$$**

| 3-gram in Candidate | Count in Candidate | Count in r1 | Count in r2 | Max(c in r1, c in r2) |
| ------------------- | ------------------ | ----------- | ----------- | --------------------- |
| "the love can"      | 1                  | 0           | 0           | 0                     |
| "love can always"   | 1                  | 1           | 0           | 1                     |
| "can always do"     | 1                  | 0           | 0           | 0                     |
| Total               | 3                  |             |             | 1                     |

$$p_3=\frac{1}{3}$$

**Calculate BP**

The length of candidate is 5, the length of $$r_1$$ is 6 and the length of $$r_2$$ is 4. Both reference sentences have the same length difference with the candidate, and in this case, we use the shorter $$r_2$$ whose length is 4. Since the length of $$r_2$$ is shorter than the length of candidate, the BP is 1.

**Calculate BLEU**

Suppose the weights is $$\lambda_1=1/3$$, $$\lambda_2=1/3$$, $$\lambda_3=1/3$$,

$$
BLEU = 1\times exp(\frac{1}{3}log(\frac{3}{5})+\frac{1}{3}log(\frac{1}{2})+\frac{1}{3}log(\frac{1}{3})) = 0.464
$$

**BLEU's output is always a number between 0 and 1**. It indicates the similarity between the candidate text or the translated text with the reference texts, **with values closer to 1 representing more similar to the references texts**. 

