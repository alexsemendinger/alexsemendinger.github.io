---
layout: post
title:  "Draft of Transformers Post"
date:   2025-02-06 13:21:13 -0500
categories: transformers
---

# A Gentle Introduction to Transformer Circuits

# DRAFT -- INCOMPLETE

# Introduction

This post introduces the Transformer architecture at a conceptual level, following the notation, language, and intuitions developed in Anthropic’s [“A Mathematical Framework for Transformer Circuits.”](https://transformer-circuits.pub/2021/framework) It builds up to an explanation of *induction heads*, a mechanism that simple Transformer models can use to perform a kind of in-context learning. I’m assuming you know a few deep learning basics: what an MLP is, something about training via SGD, **[anything else?]**, but haven’t necessarily looked into the Transformer architecture before.

My goal is to motivate the Transformer architecture so that each piece (especially the attention mechanism) feels intuitive. 

**[after draft is done: quick outline]**

# (Mostly) math-free warmup: a hypothetical language model

Before getting into the computational details of Transformers, it might help to imagine an abstract language model with the same inputs and outputs. We'll leave the mathematical details completely unspecified for now and just ask what sorts of things we'd expect such a model to do.

Here's what we know about our model:
* Inputs: a sequence of tokens $$t_1, \dots, t_n$$. Each token represents roughly one word in a sentence (for more on tokenization, see **[todo: here, link]**).
* Outputs: a probability distribution over possible next tokens in the sentence.

For example, if you input `"The Empire State Building is in New"`, a good model would assign nearly 100% probability to `"York"` and zero probability to every other token. On the other hand, for the input `"The state of New"`, the model should assign some probability to `"York"`, but also a lot to `"Jersey"`, `"Mexico"`, `"Zealand"`, `"Hampshire"`, etc.[^1]

[^1]: When I tried this on a real language model, there were also surprisingly high probabilties on `"California"` (due in part to the *Fallout* video games) and `"Austin"` (due to *Red Dead Redemption*).

**[ example sentence: “[todo – something with homography?]” ]**

## Bigrams

One model that fits this description is a lookup table of **bigram statistics**: for each pair of tokens, this tells you the frequency with which the second follows the first (say, in some large text corpus). If you throw away the information from tokens $$t_1, \dots, t_{n-1}$$ and just use $$t_n$$, this is the best you can do.

But it isn’t very good.

**[todo: example sentence(s) filled in by a bigram model. Better yet, show the top few next tokens with their probabilities.]**

You could do a bit better by looking at $$n$$\-grams for larger values of $$n$$, i.e. computing the most likely next token given the previous $$n$$ tokens. This quickly becomes impractical for a variety of reasons. To name a few: the size of the lookup table grows exponentially with $$n$$, any individual $$n$$\-gram becomes vanishingly rare in the data (it’s easy to write a 10-word phrase that has never been written before), and the model can’t use any information that appeared more than $$n$$ tokens ago.[^2]

[^2]:  Although see [https://infini-gram.io/](https://infini-gram.io/) which cleverly gets around these constraints, and which is the source of the $$n$$-gram example sentences. (But despite its impressiveness, it’s not a good autoregressive model.)

**[todo: same examples with *n*\-grams]**

## Moving information

To avoid this exponential trap, we’ll add a constraint: the model can "process" each token in some way, but the final prediction will depend solely on the processed version of the final token $$t_n$$.

There's no benefit to processing each token independently, in isolation: in that case, you still can't beat bigram statistics. So if we want to do better, we’ll have to *find a way for the other words in the context to modify the processed version of $$t_n$$*. 

The picture you should have in mind is this: whatever it means for the model to “process” a token, it should involve some representation of the information the token conveys. If a model is able to string together grammatical sentences, then some part of the model must be able to recognize whether a given token is a noun, verb, or adjective. More advanced models will need to encode much more information: which landmarks are in a given city, which movies were nominated for an Oscar, 

In order to put together grammatical sentences, the model will need to somehow encode information like the part of speech, verb tense, whether it’s singular or plural, etc. More advanced models will need to encode much more information than this: which landmarks are in a given city, which movies were nominated for an Oscar, etc, etc. We’re still talking about a purely abstract model, so we’re not making any assumptions about what this information looks like: there’s just, by necessity, some part of the model that functionally encodes something like   
```
{
    token: 'squirrel',
    position_in_sentence: 2,
    part_of_speech: 'noun',
    number: 'singular',
    likes_to_climb: 'tree',
    ...
}
```

We can describe (part of) the processing phase as “moving information between tokens.” We’ll break this down into three questions:

1. What information is being moved?  
2. For each previous token, how important is the information it’s offering?  
3. How will the information be incorporated into the representation of the current token?

Or more concisely: “What are we moving, and how are we moving it?”

**[ todo, but probably not: explore the consequences of this structure more for an abstract model? E.g.**

* **One piece of info from each prev token can move to current**  
* **Only current gets updated**  
* **Idk maybe something else ]**

# What's in a Transformer?

Summing up what we've laid out so far, we have a language model that
* takes tokens $$[t_1, \dots, t_n]$$ as inputs
* outputs a probability distribution over possible next tokens $$t_{n+1}$$
* this output is purely a function of the final "processed" version of $$t_n$$, 
* the "processing" involves somehow moving information between tokens.

These are the ingredients of a GPT-style Transformer.

## Layers and notation

Before we start, a general overview of the pieces of a Transformer and the notation we'll be using. Each of these pieces will be explained later; you can come back to this section for reference. 

**[todo: ok this doesn't work here as written. maybe appendix]**

### Activations
* Inputs: tokens $$t = [t_1, \dots, t_n]$$
    * Each $$t_i$$ is a one-hot vector of size $$n_\text{vocab}$$.
    * $$t$$ is a $$n_\text{vocab} \times n$$ matrix, with individual tokens as columns.
* Residual stream: $$x = [x_1, \dots, x_n] = W_E t.$$ 
    * Each $$x_i$$ is a vector of size $$d_\text{model}$$
    * $$x$$ is a $$d_\text{model} \times n$$ matrix, with token embeddings as columns.
* Attention head
    * $$q, k, v$$

### Parameters

# Walking through a one-layer Transformer


## Zero layers: embeddings, unembeddings, logits

Here's where the math begins.

GPT-2, which will be our running reference example, has a vocabulary size of 50,257 tokens (we'll call this $$d_{\text{vocab}}$$). Each token is represented first as a "one-hot" vector: e.g. token #324 is represented by a vector of length 50,257 consisting of all zeros except for a $$1$$ in position 324.

Fifty thousand dimensions  is a lot to work with, so the first thing the model does is **embed** the tokens into a lower-dimensional space of size $$d_{\text{model}}$$. In GPT-2, $$d_{\text{model}} = 768$$.

The simplest way to transform a vector from one dimension to another is with a single matrix multiplication, so that's what we do. For each token, we compute the embedding as $$x^{(0)} = W_E t$$, where $$W_E$$ is a matrix of size $$(d_{\text{model}}, d_{\text{vocab}})$$. We call $$W_E$$ the **embedding matrix** and $$x^{(0)}_i$$ the **embedding** of $$t_i$$.

In a trained Transformer, the embeddings already encode a significant amount of information about a token. The classic example is that $$\text{king} + \text{man} - \text{woman} \approx \text{queen}$$. This indicates that there's a sense in which some direction in the model space encodes gender, and some other direction encodes "royalty." **[todo: how justified is this to talk about dimensions]**

**[ todo: check real embeddings of gpt2 for king + man - woman example or others?]**

**[todo: move this to previous section?]** We'll see that the rest of the Transformer architecture is *permutation equivariant*: the order of tokens within a sequence doesn't matter. This would be an issue for language modeling: "the cat ran up the tree" is very different from "the tree ran up the cat." There's a simple solution: **positional embeddings**. 

The trivial “zero-layer Transformer” immediately maps these token embeddings back to a vector of size $$d_{\text{vocab}}$$ via an **unembedding matrix** $$W_U$$ of shape $$(d_\text{vocab}, d_\text{model})$$. The resulting vector $$W_U W_E t_i$$ is called the **logits**. Higher logit values correspond to likelier tokens, but this vector isn't itself a probability distribution: the entries might take any value, and don't sum to 1. To turn the logits into probabilities, we use the *softmax* function. This outputs a vector of the same size as the input with entries given by

$$ 
[\text{softmax}(x)]_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}.
$$

(The notation $$[v]_i$$ here indicates that we're taking the $$i$$th component of the vector $$v$$.)

There are several reasons this is a nice choice, but the most important is that the result can *now* be interpreted as a probability distribution, making this the end of our journey.

Expressed as a function, our zero-layer Transformer is $$T([t_1, \dots, t_n]) = T(t_n) = \text{softmax}(W_U W_E t_n)$$. The $$k$$-th entry of this output vector is the probability that the model assigns to the token with index $$k$$ appearing next.

Of course, we still haven't left the "processing tokens individually" stage, so the best we can hope for here is for the model to encode (say it with me) *bigram statistics*. And in fact, it does! **[todo: evidence?]**

So far, not very interesting -- I promised you information movement! We'll finally get that with *attention*.

## Attention part 1: updating the last token

The simplest model that deserves to be a Transformer has a layer of **attention** in between the embedding and unembedding. As a function, it looks like

$$
\begin{align*}
T(t) &= \text{softmax}(W_Ux^{(1)}_n)\\
\text{where } x^{(1)}_n &= x^{(0)}_n + [\text{Attn}(x^{(0)}_1, \dots, x^{(0)}_n)]_n.
\end{align*}
$$
**[todo: hmm dont love this... need diagram ...]**

Note that this is a **residual connection**: rather than just using the output of the attention function directly, the output is *added* to the original embedding $$x^{(0)}_n$$. Since all Transformer operations operate via residual connections, it's productive to think of them as computing adjustments to the original embedding, which flows through the network as it's processed. For this reason, we'll say the vectors $$x^{(\ell)}_i$$ at each layer $$\ell$$ are in the **residual stream**.

The actual workings of the attention function aren't so bad -- it's just a few matrix multiplications and another application of softmax -- but it's not obvious at first *why* we'd do them. So as we walk through the operations below, remember that attention is providing the "information movement" services that we want: it's computing which information should be moved, and is moving it to the right place.

The presentation here will be slightly nonstandard: in the special case of a one-layer Transformer, all we need to know is how the final token embedding $$x^{(0)}_n$$ is modified by the attention mechanism.[^3] (We'll see the standard version later, in which *every* token embedding gets modified simultaneously.)

[^3]: It's also important that we're only generating tokens from the Transformer rather than training it. **[todo: maybe explain]**

Here’s how “Attention is All You Need” summarizes attention:  

> An attention function can be described as mapping a **query** and a set of **key-value pairs** to an **output** .... The output is computed as a **weighted sum of the values**, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

We'll walk through each of these components in turn.

### Values: What information is being moved?

Our output is going to be a “weighted sum of values.” 

These values (along with the keys and queries) live in a $$d_{\text{head}}$$-dimensional space, where $$d_{\text{head}}$$ is smaller than $$d_{\text{model}}$$ (in GPT-2, it's **[todo -- also is this GPT2-small? be clear elsewhere]**). We compute the values by multiplying the embedding by a $$(d_{\text{head}}, d_{\text{model}})$$ matrix $$W_V$$: that is, $$v_i = W_V x_i^{(0)}$$.

We imagine that the embedding (somehow) represents different pieces of information in different subspaces of the residual stream. We can then think of a projection as picking out a certain subspace to use in this attention head -- that is, picking out certain information from each token to be included in our weighted sum.

Therefore, $$W_V$$ answers the question: "what information are we moving"?

### Queries and Keys: For each previous token, how important is the information in its value?

Next, we need to compute the weights. These depend on two additional parameter matrices, $$W_Q$$ and $$W_K$$, each of shape $$(d_\text{head}, d_\text{model})$$ (the same shape as $$W_V$$).

We want these weights to represent how much each previous token should inform our prediction of the next token. To figure this out, we extract some information from $$x^{(0)}_n$$, some other information from $$x^{(0)}_1, \dots, x^{(0)}_n$$, and compute a compatibility function between the two.

Concretely, we compute a **query** from the last token: $$q_n = W_Q x^{(n)}_0$$, as well as **keys** $$k_i = W_K x^{(0)}_i$$ for every token in the context (including $$x^{(0)}_n$$). The compatibility function is the dot product: $$q_n^\top k_i$$. For numerical stability reasons, you additionally divide by the square root of the head dimension, giving us **attention scores** $$s_i = q_n^\top k_i / \sqrt{d_\text{head}}$$. (We divide by $$\sqrt{d_\text{head}}$$ for numerical stability reasons.)

It's often helpful to keep activations in your neural network at roughly the same scale throughout, again for numerical stability reasons. With this in mind, we'd like the output of our weighted sum to be on the same scale as the input. One way to do that is to ensure the weights sum to 1, making the weighted sum a weighted *average*. 

We've run into this problem before, and we can use the same solution: softmax! The weights we'll use (also called the **attention pattern**) are $$a_i = [\text{softmax}([s_1, \dots, s_n])]_i = e^{s_i} / \sum_{j=1}^n e^{s_j}$$.

Putting this together, we end up with a "result" vector $$r_n = \sum_i a_i v_i$$: a weighted sum of the values, as promised.

We've now answered question 2: "for each token, how important is the information it's offering?"

### Output: How do we incorporate this information into the representation of the current token?

All that’s left is to project our weighted sum back to the residual stream. We do this via one last matrix multiplication: $$o_n = W_O r_n$$. The matrix $$W_O$$ plays a similar role to $$W_V$$, but in reverse: it picks out which subspace of the residual stream the data in $$r_n$$ will be stored in.


## Attention part 2: many queries, many heads

Our first tour through the attention mechanism described how the final token can receive information from all previous tokens in the context. In actual Transformer models, an attention head updates *every* token with information from the tokens preceding it. There are two reasons for this:
1. In a model with multiple layers of attention, this allows for *composition* of attention heads. An important example of this phenomenon is *induction heads*, which we'll see later.
2. When *training* a Transformer, the next token is predicted for each token in the sequence simultaneously: the output of ["The", "cat", "ran"] is something like ["big", "is", "quickly"] **[todo: make a better example here]**. So in contrast to the autoregressive setting, the outputs at each position are relevant.

### The attention matrix

We'll end up writing the attention mechanism somewhat differently this time around, but the only real difference is that every token will have its own query vector. The rest is bookkeeping (stacking vectors together into matrices).

Let's write this out: we compute queries, keys, and values for each token:

$$$
q = W_Q x^{(0)}, \quad k= W_K x^{(0)}, \quad v = W_V x^{(0)}.
$$$

For each query, we computute attention scores based on all the preceding keys: $$s_{ij} = q_i^\top k_j / \sqrt{d_\text{head}}$$ for $$j \leq i$$. And we turn these into weights by taking the softmax: $$a_{ij} = [\text{softmax}([s_{i1}, s_{i2}, \dots, s_{ii}])]_j$$. 

Finally, we compute our result $$r_i = \sum_{j=1}^i a_{ij} v_j$$ and our output $$o_i = W_O r_i$$, which is added to $$x^{(0)}_i$$ in the residual stream.

**[todo: ok i'm going to push thru for now but there's going to need to be some reckoning with the notation throughout. it's totally unclear that $$r_i$$ and especially $$v_i$$ are vectors while earlier $$a_i$$ is a scalar. i nearly tricked myself into thinking $$\sum_j a_{ij} v_j$$ is a matrix-vector product $$Av$$, but it isn't!]**

The double indices in $$s_{ij}$$ and $$a_{ij}$$ indicate that it might be natural to write these as matrices. And indeed, this is usually how they're presented. It makes sense to write the whole attention pattern out first (setting $$n = 4$$ so it's easy to visualize):

$$$
A = \begin{bmatrix}
a_{11} & 0 & 0 & 0 \\
a_{21} & a_{22} & 0 & 0 \\
a_{31} & a_{32} & a_{33} & 0 \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
$$$

In order to write this as a square matrix, we've set $$a_{ij} = 0$$ when $$j > i$$. This allows us to write $$r_i = \sum_{j=1}^n a_{ij} v_j$$ (earlier the sum only ran up to $$j=i$$), since the additional terms don't contribute anything.

It might not be obvious how to write the full matrix of attention *scores*. Again, we've only defined the lower-triangular portion of the matrix. But we want it to have the property that if you take the softmax of each row, you get the corresponding row of $$A$$. That is, we need to pad each row with a value that serves as an identity for softmax, the same way that $$0$$ serves as an identity for addition.

Let's look at a concrete example of softmax to figure out what this should be:

$$$
\text{softmax}([1, 2]) = \bigg[\frac{e}{e + e^2}, \frac{e^2}{e + e^2}\bigg] \approx
[0.269, 0.731]
$$$

We want to pad this with some value $$P$$ so that $$\text{softmax}([1, 2, P]) = [0.269, 0.731, 0]$$. Looking at the softmax formula, this means we want $$e^P = 0$$. The "solution" is to set $$P = -\infty$$. (In practice, you might just use a large negative value.)

So our attention score matrix is

$$$
S = \begin{bmatrix}
s_{11} & -\infty & -\infty & -\infty \\
s_{21} & s_{22} & -\infty & -\infty \\
s_{31} & s_{32} & s_{33} & -\infty \\
s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix} = \begin{bmatrix}
q_1^\top k_1 & -\infty & -\infty & -\infty \\ 
q_2^\top k_1 & q_2^\top k_2 & -\infty & -\infty \\
q_3^\top k_1 & q_3^\top k_2 & q_3^\top k_3 & -\infty \\
q_4^\top k_1 & q_4^\top k_2 & q_4^\top k_3 & q_4^\top k_4
\end{bmatrix}
$$$

We can also write this more concisely as:

$$$
q^\top k = \begin{bmatrix}
q_1^\top k_1 & q_1^\top k_2 & q_1^\top k_3 & q_1^\top k_4 \\
q_2^\top k_1 & q_2^\top k_2 & q_2^\top k_3 & q_2^\top k_4 \\
q_3^\top k_1 & q_3^\top k_2 & q_3^\top k_3 & q_3^\top k_4 \\
q_4^\top k_1 & q_4^\top k_2 & q_4^\top k_3 & q_4^\top k_4
\end{bmatrix}
$$$

With this notation, we can write the attention function the way it's presented in "Attention is All You Need":

$$$
r = \text{softmax}^*\bigg( \frac{q^\top k} {\sqrt{d_\text{head}}}\bigg)v
$$$

where $$\text{softmax}^*$$ indicates that you need to replace the upper-triangular portion of the matrix with $$-\infty$$ values to prevent information from flowing in the wrong direction.

**[todo: W_0]**

The end-to-end formula for a single attention head is therefore

$$$
x \leftarrow x + \text{softmax}^*\bigg( \frac{x^\top W_Q^\top W_K x} {\sqrt{d_\text{head}}}\bigg) x^\top W_V^\top W_O^\top.
$$$