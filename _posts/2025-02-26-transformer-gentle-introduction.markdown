---
layout: post
title:  "A Gentle Introduction to Transformers"
date:   2025-02-26 10:00:00 -0500
categories: transformers
permalink: /transformers-gentle-introduction
---

**Draft, work in progress. Feedback welcomed on [this Google Doc (equations won't display)](https://docs.google.com/document/d/1aLBG-SPTOHepFRa5gfcLGPvFx5MpWx8mo8ncjqXIQ-g/edit?usp=sharing)**.

# Introduction

The internet is blessed with an abundance of high-quality blog posts explaining how Transformers work. Some of my favorites are:

* 3blue1brown's deep learning series ([Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M&vl=en), [Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en), [MLPs](https://www.youtube.com/watch?v=9-Jl0dxWQs8&vl=en))
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
* [LLM Visualization](https://bbycroft.net/llm) by Brendan Bycroft
* [Yet Another Transformer Explainer](https://highdimensionalgrace.com/posts/big_transformer/) by Grace Proebsting

With so many options, why spend time writing one more? I find most Transformer posts are heavily focused on *what* computations are involved, walking through each tensor transformation and how you might see it in code. But they tend to say less about why you'd use *these computations in particular*. I'll be erring in the other direction, hoping to develop helpful intuitions at the cost of practical implementation advice (and brevity).

If you don't know what Transformers are yet, or if you *kind of* know what they are but couldn't say what $$\text{softmax}(QK^\top / \sqrt d)$$ is really supposed to *mean* -- why someone would decide to write that expression, or why other people use questionable language like "this represents tokens asking each other questions" to describe it -- then this post was written for you.

I'll be loosely following the notation, language, and exposition in Anthropic's [“A Mathematical Framework for Transformer Circuits”](https://transformer-circuits.pub/2021/framework) (hereafter, "Transformer Circuits").



## Post overview

In order to build up intuitions for what a Transformer model does, we'll start by abstracting away all of its inner workings and just focus on the input and output. Given *some* language model of this type, what might we *hope* it learns? And what would need to happen inside the model for this to be possible?
 
In keeping with the *gentle introduction* theme, we then walk through the attention mechanism (the most important piece of the Transformer architecture) in two stages: the first time, keeping track of how it updates *one* token, and the next looking at how it updates the rest.

We'll close by briefly discussing MLPs and LayerNorm, which are the other main components of a Transformer.

# Warmup: a hypothetical language model

(If you want to get right to Transformers, you can skip to ["What's in a Transformer"](#whats-in-a-transformer) below)

Before getting into the computational details of Transformers, it might help to imagine an abstract language model with the same inputs and outputs. We'll leave the mathematical details completely unspecified for now and just ask what sorts of things we'd expect such a model to do.

Here's what we know about our model:
* Inputs: a sequence of tokens $$t_1, \dots, t_n$$. Each token represents roughly one word in a sentence.[^1] 
* Outputs: a probability distribution over possible next tokens in the sentence.

[^1]: To get a sense for how "tokenization" works, I recommend playing with [Tiktokenizer](https://tiktokenizer.vercel.app/), which illustrates how various language models split text into tokens. For more technical details on tokenization, I recommend [this post](https://christophergs.com/blog/understanding-llm-tokenization).

For example, if you input `"The Empire State Building is in New"`, a good model would assign nearly 100% probability to `"York"` and ~zero probability to every other token. On the other hand, for the input `"The state of New"`, the model should assign some probability to `"York"`, but also a lot to `"Jersey"`, `"Mexico"`, `"Zealand"`, `"Hampshire"`, etc.[^2]

[^2]: When I tried this on a real language model, there were also surprisingly high probabilties on `"California"` (due in part to the *Fallout* video games) and `"Austin"` (due to *Red Dead Redemption*).


## Bigrams

One model that fits this description is a lookup table of **bigram statistics**: for each pair of tokens, this tells you the frequency with which the second follows the first (say, in some large text corpus). If you throw away the information from tokens $$t_1, \dots, t_{n-1}$$ and just use $$t_n$$, this is the best you can do.

Of course, this is a terrible way to generate text.

```
Prompt: Bigram language models
Output: Bigram language models. The first time. The first time.
```

You could do a bit better by looking at $$n$$\-grams for larger values of $$n$$, i.e. computing the most likely next token given the previous $$n$$ tokens. This quickly becomes impractical for a variety of reasons. To name a few: 
* The size of the lookup table grows exponentially with $$n$$.[^3]
* Any individual $$n$$\-gram becomes vanishingly rare in the data (it’s easy to write a 10-word phrase that has never been written before).
* The model can’t use any information that appeared more than $$n$$ tokens ago.

[^3]:  Although see [https://infini-gram.io/](https://infini-gram.io/) which uses some clever tricks to approximate $$n$$-grams for arbitrary values of $$n$$ (and which is the source of the $$n$$-gram example sentences). Despite its impressiveness, it’s not useful as an autoregressive model.

For situations where $$n$$-grams are appropriate, there are more sophisticated ways to work around these problems. 

## Getting information out of tokens

To avoid this exponential trap, we’ll add a constraint to our (still hypothetical) model: the model can "process" each token in some way, but the final prediction will depend solely on the processed version of the final token $$t_n$$.

There's no benefit to processing each token independently, in isolation: in that case, you still can't beat bigram statistics. So if we want to do better, we’ll have to *find a way for the other tokens in the context to modify the model's version of $$t_n$$*. 

The picture you should have in mind is this: whatever it means for the model to “process” a token, it should involve some representation of the information the token conveys. If a model is able to string together grammatical sentences, then some part of the model must be able to determine (at least implicitly) whether a given word is a noun, verb, or adjective. More advanced models will need to encode much more information: if you mention a city, it should determine what language is spoken there and what the famous landmarks are. If you mention an object, it should be able to say how big it is, whether it has a standard color, and what it's made of.

Consider the sentence
```
The strong wind might blow down the tree.
```
These are all common words, so each word is represented by a single token. But many tokens are ambiguous: 
* `wind` could mean "gust," or "wind a clock," or "knock the air out of" (as in "he was winded"). It could also appear figuratively in expressions like "caught wind of." 
* `might` indicates possibility, but it could also mean "power."
* `blow` could refer to a strike or punch, but here it refers to air moving.
* `tree` is usually a tall plant, but could be a family tree or a data structure.

In order to understand this text, you need to use context to figure out which meaning of each word is intended. You need to track that the tree is *currently* standing, but might not be for much longer. And if you want to *continue* the text, you need to have some sense of the circumstances which might give rise to such a sentence: perhaps a description of a storm will follow, or a suggestion to move something that's in danger of getting hit.

This is a lot to keep track of for such a simple sentence! But somehow, modern language models are able to do this easily.

## Moving information around

Above, I said we want to "find a way for the other tokens in the context to modify the model's version of $$t_n$$." We can rephrase this as: we want to be able to *move information* from earlier tokens to the last token.

So at least part of the "processing" done by the model should be understood as "information movement." We’ll break this down into three questions:

1. What information is being "extracted" from each token?
2. For each earlier token, how important is the information it’s offering?  
3. How will the information be incorporated into the representation of the current token?

Or more concisely: “What are we moving, and how are we moving it?”

# What's in a Transformer? <a name="whats-in-a-transformer"></a>

Summing up what we've laid out so far, we have a language model that
* Takes tokens $$[t_1, \dots, t_n]$$ as inputs.
* Outputs a probability distribution over possible next tokens $$t_{n+1}$$.
* This output is purely a function of the final "processed" version of $$t_n$$.
* The "processing" involves somehow moving information between tokens.

You shouldn't be surprised to hear that these are exactly the ingredients of a GPT-style Transformer.

# A "zero-layer Transformer": embeddings, unembeddings, logits

## Token embeddings

Here's where the math begins.

GPT-2 Small, which will be our running reference example, has a vocabulary size of 50,257 tokens (we'll call this $$n_{\text{vocab}}$$). Each token $$t_i$$ is represented first as a "one-hot" vector: e.g. the token with index -24 is represented by a vector of length 50,257 consisting of a $$1$$ in position 324 and zeros elsewhere. We'll bundle all these one-hot-encoded tokens together as the columns of an $$n_\text{vocab} \times n$$ matrix $$t = [t_1, \dots, t_n]$$, which will serve as our input.

Fifty thousand dimensions  is a lot to work with, so the first thing the model does is **embed** the tokens into a lower-dimensional space of size $$d_{\text{model}}$$. In GPT-2, $$d_{\text{model}} = 768$$.

The simplest way to transform a vector from one dimension to another is with a single matrix multiplication, so that's what we do. For each token, we compute the embedding $$W_E t$$, where $$W_E$$ is a $$d_{\text{model}} \times n_{\text{vocab}}$$ matrix. We call $$W_E$$ the **embedding matrix**.

In a trained Transformer, the embeddings already encode a significant amount of information about a token. The [classic example](https://www.technologyreview.com/2015/09/17/166211/king-man-woman-queen-the-marvelous-mathematics-of-computational-linguistics/) is that $$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$. We can think of the $$\text{king} - \text{man}$$ vector as encoding a "royalty direction" in this 768-dimensional space, such that when this direction is added to $$\text{woman}$$, we get the corresponding royal position $$\text{queen}$$.

## Positional embeddings

In addition to knowing *what* each token represents, it's important for the model to know *where in the context* the token appears. Transformers are often described as operating on sequence data, but they actually operate by default on *sets*: the order of the elements in context doesn't matter by default. That would be an issue for a language model: you can't around move words a sentence in without completely changing its meaning (or rendering it incoherent)!

There are several solutions to this problem. We'll go with the simplest, which is to use **positional embeddings**: a $$d_\text{model} \times n$$ matrix $$W_\text{pos}$$, which is added to the token embeddings $$W_E t$$. This can either be a fixed matrix ([see here for details on the encoding used in the original Transformer paper](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)) or be learned along with the rest of the model parameters. There's much more to be said about positional embeddings (not to mention other strategies for keeping track of position).

Putting the token and positional embeddings together, we have the "level-zero representation" of our tokens in the model:

$$
x^{(0)} = W_E t + W_\text{pos}.
$$

## Unembedding

The trivial “zero-layer Transformer” immediately maps these token embeddings back to a vector of size $$d_{\text{vocab}}$$ via a $$d_\text{vocab} \times d_\text{model}$$ **unembedding matrix** $$W_U$$. The entries of the resulting vector $$W_U x^{(0)}$$ are called the **logits**. Higher logit values correspond to likelier tokens, but this vector isn't itself a probability distribution: the entries might take any value, and don't sum to 1. To turn the logits into probabilities, we use the *softmax* function, defined by

$$ 
\text{softmax}(x) = \text{softmax}\left(\begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} \right) = \frac{1}{e^{x_1} + \dots + e^{x_n}} \begin{bmatrix} e^{x_1} \\ \vdots \\ e^{x_n}\end{bmatrix}.
$$

There are a number of reasons this is a nice choice, but the most important are:
1. For any value of $$x$$, $$e^x$$ is positive, so each entry in $$\text{softmax}(x)$$ is positive.
2. The sum of the entries in $$\text{softmax}(x)$$ is $$(\sum_{i=1}^n e^{x_i}) / (\sum_{i=1}^n e^{x_i}) = 1$$.

That is: for any input $$x$$, we can interpret $$\text{softmax}(x)$$ as a probability distribution, just like we wanted!

Expressed as a function, our zero-layer Transformer is 
$$
T([t_1, \dots, t_n]) = T(t_n) = \text{softmax}(W_U x^{(0)}_n) = \text{softmax}\big(W_U(W_Et_n + W_\text{pos}) \big).
$$

The $$k$$-th entry of this output vector is the probability that the model assigns to the token with index $$k$$ appearing next.

Of course, we still haven't left the "processing tokens individually" stage, so the best we can hope for here is for the model to encode (say it with me) *bigram statistics*. "Transformer Circuits" reports observing this in practice:

> In particular, the $$W_U W_E$$ term seems to often help represent bigram statistics which aren't described by more general grammatical rules, such as the fact that "Barack" is often followed by "Obama".

So far, not very interesting. I promised there would be information movement! We'll finally get that with *attention*.

# A simplified one-layer Transformer

The simplest model that deserves to be called a Transformer has a layer of **attention** in between the embedding and unembedding. 

$$
\begin{align*}
x^{(0)} &= W_E t + W_\text{pos} & \text{(embedding)} \\
x^{(1)}_n &= x^{(0)}_n + \text{Attn}(x^{(0)}_1, \dots, x^{(0)}_n) & \text{(add attention result)} \\
T(t) &= \text{softmax}(W_Ux^{(1)}_n) & \text{(unembedding)}
\end{align*}
$$

Note that this is a **residual connection**: rather than setting $$x^{(1)} = \text{Attn}(x^{(0)}_1, \dots, x^{(0)}_n)$$ directly, the attention output is *added* to the original embedding $$x^{(0)}_n$$. 

In fact, every layer of a Transformer uses residual connections. Because of this, it's helpful to imagine the original embedding $$x^{(0)}$$ "flowing through" the network, with each successive layer adding small updates to it. For this reason (following the terminology from "Transformer Circuits"), we'll say the vectors $$x^{(\ell)}_i$$ at each layer $$\ell$$ are in the **residual stream**.

The actual workings of the attention function aren't so bad -- it's just a few matrix multiplications and another application of softmax -- but it's not obvious at first *why* we'd do them. So as we walk through the operations below, remember that attention is providing the "information movement" services that we want: what information should we take from each token, how relevant is each bit to the last token, and how do we incorporate the relevant pieces of information into an updated representation of the last token?

The presentation here will be slightly nonstandard: in the special case of a one-layer Transformer, all we need to know is how the final token embedding $$x^{(0)}_n$$ is modified by the attention mechanism. (We'll see the standard version later, in which *every* token embedding gets modified simultaneously.)

Here’s how “Attention is All You Need” summarizes attention:  

> An attention function can be described as mapping a **query** and a set of **key-value pairs** to an **output** .... The output is computed as a **weighted sum of the values**, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

We'll walk through each of these components in turn.

## Values: What information is being moved?

Our output is going to be “a weighted sum of the values.” 

These values (along with the keys and queries) live in a $$d_{\text{head}}$$-dimensional space, where $$d_{\text{head}}$$ is smaller than $$d_{\text{model}}$$ (in GPT-2, it's 64, compared to $$d_\text{model} = 768$$). We compute the values by multiplying the embedding by a $$d_{\text{head}} \times d_{\text{model}}$$ matrix $$W_V$$: that is, $$v_i = W_V x_i^{(0)}$$.

We imagine that the embedding (somehow) represents different pieces of information in different subspaces of the residual stream. We can then think of a projection as picking out a certain subspace to use in this attention head -- that is, picking out certain information from each token to be included in our weighted sum.

Therefore, $$W_V$$ answers the question: "what information are we moving"?

## Queries and Keys: For each previous token, how important is the information in its value?

Next, we need to compute the weights. These depend on two additional parameter matrices, $$W_Q$$ and $$W_K$$, each of shape $$(d_\text{head}, d_\text{model})$$ (the same shape as $$W_V$$).

We want these weights to represent how much each previous token should inform our prediction of the next token. To figure this out, we extract some information from $$x^{(0)}_n$$, some other information from $$x^{(0)}_1, \dots, x^{(0)}_n$$, and compute a compatibility function between the two.

Concretely, we compute a **query** from the last token: $$q_n = W_Q x^{(n)}_0$$, as well as **keys** $$k_i = W_K x^{(0)}_i$$ for every token in the context (including $$x^{(0)}_n$$). The compatibility function is the dot product: $$q_n^\top k_i$$. For numerical stability reasons, you additionally divide by the square root of the head dimension, giving us **attention scores** $$s_i = q_n^\top k_i / \sqrt{d_\text{head}}$$.

(Why divide by $$\sqrt{d_\text{head}}$$? The short answer is: it's often helpful to keep activations in your neural network at roughly the same scale throughout, and this turns out to be the right scaling value. The semi-formal argument for this is that if the entries of $$q, k$$ are independent random variables with mean $$0$$ and variance $$1$$, then $$q^\top k$$ has mean $$0$$ and variance $$d_\text{head}$$. That means you'll commonly see much larger values! But $$q^\top k / \sqrt{d_\text{head}}$$ has mean $$0$$ and variance $$1$$, which is "on the same scale" as $$q$$ and $$k$$.)

In keeping with the idea of keeping activations on the same scale, we'd also like the output of our weighted sum to be on the same scale as the input. One way to do that is to ensure the weights sum to $$1$$, making the weighted sum a weighted *average*. 

Luckily, we already know a function that does just this: softmax! So the weights we'll use (also called the **attention pattern**) are $$a_i = \text{softmax}([s_1, \dots, s_n])$$.

Putting this together, we end up with a "result" vector $$r_n = \sum_i a_i v_i$$: a weighted sum of the values, as promised.

We've now answered question 2: "for each token, how important is the information it's offering?"

## Output: How do we incorporate this information into the representation of the current token?

All that’s left is to project our weighted sum back to the residual stream. We do this via one last matrix multiplication: $$o_n = W_O r_n$$. The matrix $$W_O$$ plays a similar role to $$W_V$$, but in reverse: it picks out which subspace of the residual stream the data in $$r_n$$ will be stored in.

This gets added to the orignal token embedding: $$x^{(1)}_n = x^{(0)}_n + o_n$$.

## Unembedding, logits, probabilities

We've reached the end of the residual stream in our tiny one-layer model, so it's time to compute the outputs. We unembed to produce logits $$\ell_{n+1} = W_U x^{(1)}_n$$, which we can then turn into probabiltiies $$p_{n+1} = \text{softmax}(\ell_{n+1})$$. Each of these is a vector of size $$n_\text{vocab}$$, with entries reflecting the probability the model assigns to each possible next token following the input sequence.


## The full one-layer, attention-only, just-predicts-the-next-token Transformer

To sum up, here are all the parameters and activations of our simplified one-layer Transformer. Remember that this version is nonstandard: if you want to compare this to a practical Transformer implementation, you should use the tables that appear later.

| Activation Name   | Expression                                   | Shape                     |
|-------------------|----------------------------------------------|---------------------------|
| Input tokens      | $$t = [t_1, \dots, t_n]$$                    | $$n_\text{vocab} \times n$$ |
| Embedding         | $$x = [x_1, \dots, x_n] = W_E t + W_\text{pos}$$ | $$d_\text{model} \times n$$ |
| Query             | $$q_n = W_Q x_n$$                            | $$d_\text{head} \times 1$$ |
| Keys              | $$k = [k_1, \dots, k_n]= W_K x$$             | $$d_\text{head} \times n$$ |
| Values            | $$v = [v_1, \dots, v_n] = W_V x$$            | $$d_\text{head} \times n$$ |
| Attention scores  | $$s = q_n^\top k / \sqrt{d_\text{head}}$$    | $$n$$  |
| Attention weights | $$a = \text{softmax}(s)$$                    | $$n$$  |
| Attention result  | $$r_n = \sum_{i=1}^n a_i v_i$$               | $$d_\text{head} \times 1$$ |
| Attention output  | $$o_n = W_O r_n$$                            | $$d_\text{model}\times 1$$ |
| Updated last-token embedding | $$x^{(1)}_n = x_n + o_n$$         | $$d_\text{model}\times 1$$ |
| Next-token logits | $$\ell_{n+1} = W_U x^{(1)}_n$$               | $$n_\text{vocab}\times 1$$ |
| Next-token probabilities | $$p_{n+1} = \text{softmax}(\ell_n)$$  | $$n_\text{vocab}\times 1$$ |


| Parameter Name | Shape                       |
|----------------|-----------------------------|
| $$W_E$$        | $$d_\text{model} \times n_\text{vocab}$$ |
| $$W_\text{pos}$$ | $$d_\text{model} \times n$$ |
| $$W_Q, W_K, W_V$$| $$d_\text{head} \times d_\text{model}$$ |
| $$W_O$$        | $$d_\text{model} \times d_\text{head}$$ |
| $$W_U$$        | $$n_\text{vocab} \times d_\text{model}$$ |

# The full one-layer Transformer

Our first tour through the attention mechanism described how the *final* token can receive information from all previous tokens in the context. In actual Transformer models, an attention head updates *every* token with information from the tokens preceding it. There are two reasons for this:
1. In a model with multiple layers of attention, this allows information to take multiple hops between tokens, allowing for richer contextual representatinos and more expressive algorithms. One important algorithm of this type is the [induction head](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), which performs a simple kind of in-context learning.
2. When *training* a Transformer, next-token probabilities are computed for every position simultaneously and compared to the actual next tokens that appear in the sequence. This is a huge efficiency improvement: when processing a sequence with $$n$$ tokens, this means you get $$n$$ pieces of feedback rather than just one.

In this walkthrough, we'll rewrite the attention mechanism in a way that reflects this, and also add in the second (simpler) piece of a Transformer block: the multilayer perceptron (MLP) layer.

## The attention *matrix*

We'll end up writing the attention mechanism somewhat differently this time around, but the only real difference is that every token will have its own query vector. The rest is bookkeeping (stacking vectors together into matrices).

Let's write this out: we compute queries, keys, and values for each token:

$$
q = W_Q x^{(0)}, \quad k= W_K x^{(0)}, \quad v = W_V x^{(0)}.
$$

For each query, we computute attention scores based on all the preceding keys: $$s_{ij} = q_i^\top k_j / \sqrt{d_\text{head}}$$ for $$j \leq i$$. And we turn these into weights by taking the softmax: $$a_{i} = \text{softmax}([s_{i1}, s_{i2}, \dots, s_{ii}])$$. (Note that $$a_i$$ is now an $$i$$-dimensional vector, with components $$a_{ij}$$.)

Finally, we compute our result $$r_i = \sum_{j=1}^i a_{ij} v_j$$ and our output $$o_i = W_O r_i$$, which is added to $$x^{(0)}_i$$ in the residual stream.

The double indices in $$s_{ij}$$ and $$a_{ij}$$ indicate that it might be natural to write these as matrices. And indeed, this is usually how they're presented. It makes sense to write the whole attention pattern out first (setting $$n = 4$$ so it's easy to visualize):

$$
A = \begin{bmatrix}
a_{11} & 0 & 0 & 0 \\
a_{21} & a_{22} & 0 & 0 \\
a_{31} & a_{32} & a_{33} & 0 \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
$$

In order to write this as a square matrix, we've set $$a_{ij} = 0$$ when $$j > i$$. This allows us to write $$r_i = \sum_{j=1}^n a_{ij} v_j$$ (summing up to $$j=n$$ rather than stopping at $$j=i$$), since the additional terms don't contribute anything.

It might not be obvious how to write the full matrix of attention *scores*. Again, we've only defined the lower-triangular portion of the matrix. But we want it to have the property that if you take the softmax of each row, you get the corresponding row of $$A$$. That is, we need to pad each row with a value that serves as an identity for softmax, the same way that $$0$$ serves as an identity for addition.

Let's look at a concrete example of softmax to figure out what this should be:

$$
\text{softmax}([1, 2]) = \bigg[\frac{e}{e + e^2}, \frac{e^2}{e + e^2}\bigg] \approx
[0.269, 0.731]
$$

We want to pad this with some value $$P$$ so that $$\text{softmax}([1, 2, P]) = [0.269, 0.731, 0]$$. Looking at the softmax formula, this means we want $$e^P = 0$$. The "solution" is to set $$P = -\infty$$. (In practice, you might just use a large negative value.)

So our attention score matrix is

$$
S = \begin{bmatrix}
s_{11} & -\infty & -\infty & -\infty \\
s_{21} & s_{22} & -\infty & -\infty \\
s_{31} & s_{32} & s_{33} & -\infty \\
s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix} = \frac{1}{\sqrt{d_\text{head}}}\begin{bmatrix}
q_1^\top k_1 & -\infty & -\infty & -\infty \\ 
q_2^\top k_1 & q_2^\top k_2 & -\infty & -\infty \\
q_3^\top k_1 & q_3^\top k_2 & q_3^\top k_3 & -\infty \\
q_4^\top k_1 & q_4^\top k_2 & q_4^\top k_3 & q_4^\top k_4
\end{bmatrix}
$$

We can also write this more concisely as:

$$
q^\top k = \begin{bmatrix}
q_1^\top k_1 & q_1^\top k_2 & q_1^\top k_3 & q_1^\top k_4 \\
q_2^\top k_1 & q_2^\top k_2 & q_2^\top k_3 & q_2^\top k_4 \\
q_3^\top k_1 & q_3^\top k_2 & q_3^\top k_3 & q_3^\top k_4 \\
q_4^\top k_1 & q_4^\top k_2 & q_4^\top k_3 & q_4^\top k_4
\end{bmatrix}
$$

which lets us write the attention pattern as

$$
A = \text{softmax}^*\bigg( \frac{q^\top k} {\sqrt{d_\text{head}}}\bigg)
$$

where $$\text{softmax}^*$$ indicates that you need to replace the upper-triangular portion of the matrix with $$-\infty$$ values to prevent information from flowing in the wrong direction.

We can package our result calculations $$r_i = \sum_{j=1}^n a_{ij} v_j$$ for $$i=1, \dots, n$$ into one matrix-vector product: $$r = vA^\top$$, and then project back to the residual stream via $$o = W_O r$$.

## Multiple heads

Up to this point, I've been acting as if there's a single attention calculation in each attention block. But in practice, this isn't the case: attention blocks will have many "heads" of attention running in parallel. (In GPT-2 Small, each attention layer has 12 attention heads.) Each head $$h_i$$ has its own weight matrices $$W_Q^{h_i}, W_K^{h_i}, W_V^{h_i}$$ producing queries, keys, and values $$q^{h_i}, k^{h_i}, v^{h_i}$$, attention pattern $$A^{h_i}$$, and results $$r^{h_i}$$. Typically, if there are $$H$$ heads, the head dimension will be $$d_\text{head} = d_\text{model} / H$$.

There are two equivalent ways to think about how to combine the results of each attention head. The conceptually simpler way, used in the "Transformer Circuits" paper, is to give each attention head its own output matrix $$W_O^{h_i}$$ and add up the outputs of each head: $$x^{(1)} = x^{(0)} + \sum_{i=1}^H o^{h_i}$$. This makes it clear that each head operates independently, and each contributes to the result in exactly the same way.

However, this *isn't* how the orignal paper on Transformers writes the operation or how it's usually implemented. Matrix multiplication is a highly optimized operation, making it more efficient to perform one big matrix multiplication rather than adding up the results of several small matrix multiplications.

Here we let $$r^{h_1}, \dots, r^{h_H}$$ be the results from each attention head, and let

$$
R = \begin{bmatrix} r^{h_1} \\ \vdots \\ r^{h_H} \end{bmatrix}
$$

be the vector of size $$d_\text{head} \cdot H = d_\text{model}$$ obtained from stacking them on top of each other.  The overall attention output is then $$o = W_O R$$, where $$W_O$$ is $$d_\text{model} \times d_\text{model}$$. (Note that we're now *enforcing* the identity $$d_\text{head} = d_\text{model} / H$$, whereas this was just a convention from the additive perspective.)

Why are these the same? We can split up $$W_O$$ into a block matrix $$[W_O^{h_1} \,\vert\, \dots \,\vert\, W_O^{h_H}]$$, where each block is of shape $$d_\text{model} \times d_\text{head}$$. Then

$$
W_O R = \left[W_O^{h_1} \,|\, \dots \,|\, W_O^{h_H}\right] \begin{bmatrix} r^{h_1} \\ \vdots \\ r^{h_H} \end{bmatrix} = \sum_{i=1}^H W_O^{h_i} r^{h_i}.
$$

Going forward, we'll stick with the "independent and additive" interpretation, following "Transformer Circuits." But it's important to remember that this *isn't* what you'll see in a typical Transformer implementation.

The end-to-end formula for a full layer of attention is therefore

$$
x^{(1)} = x^{(0)} + \sum_{h=1}^{H} W_O^h W_V^h\,   x^{(0)} \, \text{softmax}^*\bigg( \frac{(x^{(0)})^\top (W_Q^h)^\top W_K^h x^{(0)}} {\sqrt{d_\text{head}}}\bigg)^\top.
$$

## The multilayer perceptron

Let's take a quick step back to the abstract language model we started with. We concluded that, no matter how much processing you do to individual tokens, a model with the same inputs and outputs as a Transformer can't outperform bigram statistics (a very low bar!) unless there's a way to move information between tokens. 

But now that we've successfully moved information between tokens, we *can* benefit from some per-token processing! We do this with the simplest nontrivial neural network component, called a "fully connected layer" or "multilayer perceptron": two matrix multiplications with an elementwise nonlinear operation[^5] in the middle.

[^5]: $$\text{ReLU}(x) = \max(x, 0)$$ is a common nonlinearity for MLPs, but Transformers more commonly use the "Gaussian error linear unit" or GELU, defined as $$\text{GELU}(x) = x \cdot \Phi(x)$$ where $$\Phi(x)$$ is the cumulative distribution function of the standard Gaussian. You can [read more about GELU here](https://simplicityissota.substack.com/p/the-rise-of-gelu); the distinction is too subtle to be important for our discussion.

We start by projecting to a higher dimension $$d_\text{mlp}$$. By convention, $$d_\text{mlp} = 4 \cdot d_\text{model}$$, but there's no special reason to use this value rather than another.

$$
z = \text{ReLU}(W_1 x^{(1)} + b_1)
$$

Here, $$W_1$$ is a $$d_\text{mlp} \times d_\text{model}$$ matrix and $$b_1$$ is a $$d_\text{mlp} \times 1$$ bias vector. Each token position is independently multiplied by $$W_1$$: that is, $$W_1 x^{(1)} = [W_1 x^{(1)}_1, \dots, W_1 x^{(1)}_n]$$.

We then project back down to $$d_\text{model}$$ and add the result to the residual stream:

$$
\begin{align*}
m &= W_2 z + b_2 \\
x^{(2)} &= x^{(1)} + m.
\end{align*}
$$

The MLP is mathematically simple but conceptually more opaque. There's a clear story about how attention moves information between tokens, but the MLP's role is harder to grasp.

One strong hypothesis is that MLPs act as the model's "fact storage." Somehow, the model is able to say that Abraham Lincoln was the 16th president and the location of the world's largest ball of paint, even if those facts aren't included in the context.

```
Prompt: Where is the world's largest ball of paint? Answer with just a location.

Llama-3-70b: Alexandria, Indiana
```

If attention only moves information around, then we'd guess the MLPs are where new information like this gets added! For an overview of one way this could work, I recommend 3blue1brown's video ["How LLMs store facts."](https://www.youtube.com/watch?v=9-Jl0dxWQs8&vl=en)

## LayerNorm

There's one last component in a Transformer, which I've so far left out because it's not very *conceptually* important. (It is *practically* important! It just doesn't add much to our story of "how information is moving around.")

At a few points, we've seen that it's useful to keep activations "on the same scale" throughout the network. This explains both the $$\sqrt{d_\text{head}}$$ factor and the softmax when computing the attention pattern, for instance. LayerNorm is similar, ensuring that the inputs to each attention and MLP layer are a consistent size.

For each vector $$x_i$$ in the residual stream, we subtract the mean $$\mu(x_i)$$ and divide by the standard deviation $$\sigma(x_i)$$ of the entries. The elements of the resulting vector have mean $$0$$ and variance $$1$$. We then shift and scale to produce a vector whose entries have mean $$\beta$$ and standard deviation $$\gamma$$, where $$\beta, \gamma$$ are learned parameters of size $$d_\text{model}$$, similar to model weights.

To sum up, the LayerNorm operation is

$$
\text{LayerNorm}(x) = \frac{x - \mu(x)}{\sigma(x)} \odot \gamma + \beta.
$$

where 
* $$\odot$$ denotes element-by-element multiplication of two vectors
* $$\mu(x), \sigma(x)$$ are *scalars* representing the mean and standard deviation of the entries of $$x$$
* $$x - \mu(x)$$ is a slight abuse of notation meaning "subtract the scalar $$\mu(x)$$ from each entry of the vector $$x$$."

Empirically, LayerNorm seems to speed up training and might have other performance benefits. As with most numerical optimizations in neural networks, there are alternatives that also work well (such as "RMSNorm"), and there's little rigorous understanding of *why* popular methods work so well.


## Summing up

A full Transformer model consists of an embedding, several Transformer blocks, and an unembedding.

Token IDs, encoded as one-hot vectors, are turned into word embeddings. These embedding vectors capture the semantic information present in each individual token.

The embeddings then pass through several Transformer blocks. The attention heads allow each token to ask questions of the preceding tokens. Based on the "answers" to these questions, information from the preceding tokens will flow forward and be incorporated into an updated embedding in the residual stream. In the phrase "The Empire State Building," `Building` will be updated to represent the Empire State Building in particular. In "the thorny red flower," the `flower` token will updated to reflect its redness and thorns. Then comes another LayerNorm and the MLP, which modifies information independently. This adds extra information that is encoded in model parameters but *not* in the context: `Building` could get updated to indicate that it's in New York, and the thorny red `flower` could be updated toward "rose."

Over the course of several Transformer blocks, these word embeddings come to more richly encode information relevant to predicting what token should follow them. After the unembedding layer and a softmax, they reflect a probability distribution over all possible choices of next tokens.

## The full one-layer Transformer

To close, here's an updated version of our earlier chart of Transformer activations and parameters, reflecting an attention layer that updates *every* token and an MLP layer.

| Activation Name   | Expression                                       | Shape                     |
|-------------------|--------------------------------------------------|---------------------------|
| Input tokens      | $$t = [t_1, \dots, t_n]$$                        | $$n_\text{vocab} \times n$$ |
| Embedding | $$x^{(0)} = [x^{(0)}_1, \dots, x^{(0)}_n] = W_E t + W_\text{pos}$$ | $$d_\text{model} \times n$$ |
| LayerNorm         | $$x^{(0)}_\text{LN}=\text{LayerNorm}(x^{(0)})$$  | $$d_\text{model} \times n$$ | 
| Queries           | $$q^h = W_Q^h x^{(0)}_\text{LN} $$               | $$d_\text{head} \times n$$ |
| Keys              | $$k^h = W_K^h x^{(0)}_\text{LN}$$                | $$d_\text{head} \times n$$ |
| Values            | $$v^h = W_V^h x^{(0)}_\text{LN}$$                | $$d_\text{head} \times n$$ |
| Attention scores  | $$S^h = (q^h)^\top k^h / \sqrt{d_\text{head}}$$  | $$n \times n$$  |
| Attention weights | $$A^h = \text{softmax}^*(S^h)$$                  | $$n \times n$$  |
| Attention result  | $$r^h = v^h(A^h)^\top$$                          | $$d_\text{head} \times n$$ |
| Attention output  | $$o^h = W_O^h r^h$$                              | $$d_\text{model} \times n$$ |
| Post-attention embeddings | $$x^{(1)} = x^{(0)} + \sum_h o^h$$       | $$d_\text{model} \times n$$ |
| LayerNorm         | $$x^{(1)}_\text{LN}=\text{LayerNorm}(x^{(1)})$$  | $$d_\text{model} \times n$$ | 
| MLP hidden layer  | $$z = \text{ReLU}(W_1 x^{(1)}_\text{LN}  + b_1)$$| $$d_\text{mlp} \times n$$ | 
| MLP output        | $$m = W_2 z + b_2$$                              | $$d_\text{model} \times n$$ |
| Post-MLP embeddings | $$x^{(2)} = x^{(1)} + m$$                      | $$d_\text{model} \times n$$ |
| Logits            | $$\ell = W_U x^{(2)}$$                           | $$n_\text{vocab} \times n$$ |
| Probabilities     | $$p = \text{softmax}(\ell)$$                     | $$n_\text{vocab} \times n$$ |


| Layer          | Parameter(s)             | Shape                                           |
|----------------|--------------------------|-------------------------------------------------|
| Embedding      | $$W_E$$                  | $$d_\text{model} \times n_\text{vocab}$$        |
| Embedding      | $$W_\text{pos}$$         | $$d_\text{model} \times n$$                     |
| LayerNorm      | $$\gamma, \beta$$        | $$d_\text{model}$$                              |
| Attention Head | $$W_Q^h, W_K^h, W_V^h$$  | $$d_\text{head} \times d_\text{model}$$         |
| Attention Head | $$W_O^h$$                | $$d_\text{model} \times d_\text{head}$$         |
| MLP            | $$W_1, b_1$$             | $$d_\text{mlp} \times d_\text{model}$$, &nbsp; $$d_\text{mlp}$$  |
| MLP            | $$W_2, b_2$$             | $$d_\text{model} \times d_\text{mlp}$$, &nbsp; $$d_\text{model}$$ |
| Unembedding    | $$W_U$$                  | $$n_\text{vocab} \times d_\text{model}$$        |


# Appendix: Key figures in several LLMs 

| Parameter           | GPT-2 Small (2019) | GPT-2 XL (2019) | GPT-3 (2020) | DeepSeek V3 (2024) |
|---------------------|-----------|----------|-------------|-------------|
| **Total Parameters**| 124M      | 1.5B     | 175B        | 671B        |
| $$d_\text{model}$$  | 768       | 1600     | 12288       | 7168        |
| $$d_\text{mlp}$$    | 3072      | 6400     | 49152       | 18432       |
| $$H = n_\text{heads}$$  | 12    | 25       | 96          | 128         |
| $$d_\text{head}$$   | 64        | 64       | 128         | 56          |
| $$n_\text{layers}$$ | 12        | 48       | 96          | 61          |
| $$n_\text{vocab}$$  | 5025      | 50257    | 50257       | 129280      |
| Context length      | 1024      | 2048     | 2048        | 163840      |
