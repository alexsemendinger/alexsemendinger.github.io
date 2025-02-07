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

# A hypothetical language model

Before getting into the computational details of Transformers, it might help to imagine an abstract language model with the same inputs and outputs. 
* Input: a sequence of tokens $$t_1, \dots, t_n$$. Each token represents roughly one word in a sentence (for more on tokenization, see **[todo: here, link]**).
* Output: a probability distribution over possible next tokens in the sentence.

For example, if you input "The Empire State Building is in New", a good model would assign nearly 100% probability to "York" and zero probability to every other token. For the input "The state of New", the model should assign some probability to "York", as well as "Jersey", "Mexico", "Zealand", "Hampshire", etc.[^1]

[^1]: When I tried this on a real language model, there were also surprisingly high probabilties on "California" (due to a secessionist movement) and "Austin" (New Austin is a fictional state in Red Dead Redemption).

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

# Transformers

Summing up what we've laid out so far, we have a language model that
* takes tokens $$[t_1, \dots, t_n]$$ as inputs
* outputs a probability distribution over possible next tokens $$t_{n+1}$$
* this output is purely a function of the final "processed" version of $$t_n$$, 
* the "processing" involves somehow moving information between tokens.

These are the ingredients of a GPT-style Transformer.

## Zero layers: embeddings, unembeddings, logits

Here's where the math begins.

GPT-2, which will be our running reference example, has a vocabulary size of 50,257 tokens (we'll call this $$d_{\text{vocab}}$$). Each token is represented first as a "one-hot" vector: e.g. token #324 is represented by a vector of length 50,257 consisting of all zeros except for a $$1$$ in position 324.

Fifty thousand dimensions  is a lot to work with, so the first thing the model does is **embed** the tokens into a lower-dimensional space of size $$d_{\text{model}}$$. In GPT-2, $$d_{\text{model}} = 768$$.

The simplest way to transform a vector from one dimension to another is with a single matrix multiplication, so that's what we do. For each token, we compute the embedding as $$x^{(0)}_i = W_E t_i$$, where $$W_E$$ is a matrix of size $$(d_{\text{model}}, d_{\text{vocab}})$$. We call $$W_E$$ the **embedding matrix** and $$x^{(0)}_i$$ the **embedding** of $$t_i$$.

In a trained Transformer, the embeddings already encode a significant amount of information about a token. The classic example is that $$\text{king} + \text{man} - \text{woman} \approx \text{queen}$$: some direction in the model space seems to encode gender, and some other direction seems to encode "royalty." **[todo: how justified is this to talk about dimensions]**


**[ todo: check real embeddings of gpt2 for king + man - woman example or others?]**

The trivial “zero-layer Transformer” immediately maps these token embeddings back to a vector of size $$d_{\text{vocab}}$$ via an **unembedding matrix** $$W_U$$ of shape $$(d_\text{vocab}, d_\text{model})$$. The resulting vector $$W_U W_E t_i$$ is called the **logits**. Higher logit values correspond to likelier tokens, but this vector isn't itself a probability distribution: the entries might take any value, and don't sum to 1. To turn the logits into probabilities, we use the *softmax* function:

$$ 
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}.
$$

There are several reasons this is a nice choice, but the most important is that the result can *now* be interpreted as a probability distribution, making this the end of our journey.

Expressed as a function, our zero-layer Transformer is $$T([t_1, \dots, t_n]) = T(t_n) = \text{softmax}(W_U W_E t_n)$$. The $$k$$-th entry of this output vector is the probability that the model assigns to the token with index $$k$$ appearing next.

Of course, we still haven't left the "processing tokens individually" stage, so the best we can hope for here is for the model to encode (say it with me) *bigram statistics*. And in fact, it does! **[todo: evidence?]**

So far, not very interesting -- I promised you information movement! We'll finally get that with *attention*.

## Attention heads

The simplest model that deserves to be a Transformer has a layer of **attention** in between the embedding and unembedding. As a function, it looks like

$$
\begin{align*}
p(t_{n+1} | t_1, \dots, t_n) = \text{softmax}(W_Ux^{(1)}_n)\\
\text{where } x^{(1)}_n = x^{(0)}_n + \text{Attn}(x^{(0)}_1, \dots, x^{(0)}_n)_n.
\end{align*}
$$
**[todo: hmm dont love this... need diagram ...]**
**[todo: notation, throughout: i was using $$T(t_1, \dots, t_n)$$, but maybe $$p(t_{n+1} | t_1, \dots, t_n)$$ is better. might be less accessible though?]**

Note that this is a **residual connection**: rather than just using the output of the attention function directly, the output is *added* to the original embedding $$x^{(0)}_n$$. Since all Transformer operations operate via residual connections, it's productive to think of them as computing adjustments to the original embedding, which flows through the network as it's processed. For this reason, we'll say the vectors $$x^{(\ell)}_i$$ at each layer $$\ell$$ are in the **residual stream**.

The actual workings of the attention function aren't so bad -- it's just a few matrix multiplications and another application of softmax -- but it's not obvious at first *why* we'd do them. So as we walk through the operations below, remember that attention is providing the "information movement" services that we want: it's computing which information should be moved, and is moving it to the right place.

The presentation here will be slightly nonstandard: for a one-layer attention-only Transformer, all we need to know is how $$x^{(0)}_n$$ is modified by the attention mechanism, so we'll ignore everything else for now.[^3] Later, we'll see that every other token is also modified simultaneously

[^3]: It's also important that we're only generating tokens from the Transformer rather than training it. **[todo: maybe explain]**

Here’s how “Attention is All You Need” summarizes the mechanism:  
\> “weighted sum of the values etc..”

### Values: What information is being moved?

Our output is going to be a “weighted sum of values.” The values are smaller-dimensional vectors containing the information important to this particular attention head, obtained by projecting token embeddings from the residual stream. We project via a `(d_head, d_model)` matrix called \(W_V\).

So for each previous token embedding \(x_i\), we now have a *value vector* \(v_i = W_v x_i\). **[is multiplying on the left ok here? double check at end for consistency]**

### Queries and Keys: For each previous token, how important is the information in its value?

Next, we need to compute the weights. These depend on two additional parameter matrices, \(W_Q\) and \(W_K\), each of shape `(d_head, d_model)` (just like \(W_V\)). The current token \(x_n\) is projected to a *query* \(q_n = W_Q x_n\). As with all projections in the attention mechanism, we think of this as extracting some information from the residual stream representation \(x_i\).

### Output: How do we incorporate this information into the representation of the current token?

All that’s left is to project our weighted sum to a vector that’s the same size as \(x_n\). We do this via one last matrix multiplication: \(o_n = W_O x_n\). The matrix \(W_O\) picks out which subspace of the residual stream the data in \(r_n\) will be stored in.

### Putting it together: QK and OV circuits

We can now write the entire attention operation at once:  
**[ equation ]**

Notice that this involves two matrices of shape `(d_model, d_model)`: \(W_{QK} = W_Q^\top W_K\) and \(W_{OV} = W_O W_V\). *Math Framework* refers to these as the “QK circuit” and “OV circuit.” 

**[ interpret these matrices ]**

This structure limits the expressivity of attention heads in a subtle way. The matrix \(W_{OV}\) determines the read/write operations for all tokens simultaneously: you can’t take some information from one token and different information from the next. Similarly, \(W_{QK}\) works the same for all positions: **[i cant finish this sentence properly]**.

This leads to some bugs in one-layer attention-only Transformers.

## MLPs in Transformers

Now that we’ve moved information between tokens, we can do some “per-token processing.” We’ll use the simplest neural network component there is, the MLP.

**[describe operation]**

Attention-only Transformers are powerful enough to perform some algorithmic tasks **[link]**. But “MLP-only Transformers” would be completely useless: as we saw at the very start, no amount of per-token processing can improve on bigram statistics unless there’s also a way to move information between tokens.

**[some links to “key-value lookup” and 3b1b video on transformer MLPs if you want more intuition]**

# Stacking Layers

## Two Layer Attention-Only Transformer

* Attention *matrix*

## Induction Heads

