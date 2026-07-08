---
layout: post
title:  "How do you tell apart keys and queries in Transformers?"
date:   2025-04-17 12:00:00 -0500
categories: transformers
permalink: /transformers-key-query
---

<!-- 
Two ways that *work* to include images.
(The default copy-paste doesn't work.)
Note that you should put them in /assets/images

![alt text](/assets/images/image.png)

<figure class="image-container">
  <img src="{{ site.baseurl }}/assets/images/image.png" alt="Image description" class="responsive-image">
  <figcaption>Your image caption here</figcaption>
</figure> 
-->

# TL;DR

* The *query* comes from the "final token in the sequence," i.e. the one that immediately precedes the "next token" that the Transformer is predicting.

* The *keys* come from all previous tokens. 

* This is easy to see when considering the case of *inference on a one-layer Transformer*: every token's *key* is used in the output, but only the final token's *query* is used.
    * Therefore, we can consider a simpler, equivalent model in which only the final token has a query.

If this made sense, congratulations, you don't need to read the rest of the post! Otherwise, I'll explain everything below.

# Introduction

The most valuable insight I had when writing my [Transformer explainer post](https://alexsemendinger.github.io/transformers-gentle-introduction) was how to reliably remember the difference between keys and queries in Transformer attention blocks. Standard formulas like

$$
\text{Attention}(Q, K, V) = \text{softmax}\bigg( \frac{QK^\top} {\sqrt{d_\text{head}}} \bigg) V
$$

put the queries and keys on a fairly equal footing. But you need to remember that the softmax is over the ... key dimension? Right? And when generating text you should use a KV cache, but why isn't that a QV cache? What's going on there?

An easy way to see the difference is to consider a *one-layer Transformer*, modified in a way that is computationally equivalent to the usual implementation, but in which queries and keys play very obviously distinct roles.

# A simplified one-layer Transformer

The key insight: suppose you pass tokens $$[t_1, \dots, t_n]$$ through a Transformer. The tokens pass through an embedding, producting the initial residual stream vectors $$[x_1^{(0)}, \dots, x_n^{(0)}]$$. They pass through several Transformer blocks, resulting in the final residual stream vector $$[x_1^{(L)}, \dots, x_n^{(L)}]$$. Finally, and obtain output logits $$[\ell_2, \dots, \ell_{n+1}]$$ (I've re-indexed so that $$\ell_i$$ is the logit vector for the $$i$$-th token position).

$$
\begin{align*}
x^{(0)} &= W_E t + W_\text{pos} & \text{(embedding)} \\
x^{(1)}_n &= x^{(0)}_n + \text{Attention}(x^{(0)}_1, \dots, x^{(0)}_n) & \text{(add attention result)} \\
T(t) &= \text{softmax}(W_Ux^{(1)}_n) & \text{(unembedding)}
\end{align*}
$$

![A modified version of the previous diagram: now, after tokens are embedded, an "attention output" o is produced from x^0. x^0 and o are combined to produce x^1, which is unembedded to produce logits and probabilities. Together, x^0 and x^1 make up the "residual stream."](/assets/images/tf1-onelayersimple-overview.png)


**[END OF TODO: PROOFREAD, REWRITTEN]**

The actual workings of the attention function aren't so bad -- it's just a few matrix multiplications and another application of softmax -- but it's not obvious at first *why* we'd do them. So as we walk through the operations below, remember that attention is providing the "information movement" services that we want: what information should we take from each token, how relevant is each bit to the last token, and how do we incorporate the relevant pieces of information into an updated representation of the last token?

Here’s how “Attention is All You Need” summarizes attention:  

> An attention function can be described as mapping a **query** and a set of **key-value pairs** to an **output** .... The output is computed as a **weighted sum of the values**, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

**[TODO: incorporate this better]**

Here's a diagram illustrating the attention mechanism, which we'll walk through piece by piece.

![A diagram illustraing the simplified attention mechanism which only updates the last token embedding. Every token has an associated key and value, and the final token has an associated query. The keys combine with the query to compute scores, which result in attention weights after a softmax operation. A weighted sum of the values is computed, using the attention weights. This result is projected back into the residual stream to produce the attention output.](/assets/images/tf1-onelayersimple-attention.png)

## Values: what information is being moved?

Our output is going to be “a weighted sum of the values.” 

These values (along with the keys and queries) live in a $$d_{\text{head}}$$-dimensional space, where $$d_{\text{head}}$$ is smaller than $$d_{\text{model}}$$ (in GPT-2, it's 64, compared to $$d_\text{model} = 768$$). We compute the values by multiplying the embedding by a $$d_{\text{head}} \times d_{\text{model}}$$ matrix $$W_V$$. That is, $$v_i = W_V x_i^{(0)}$$.

We imagine that the embedding (somehow) represents different pieces of information in different subspaces of the residual stream. We can then think of a projection as picking out a certain subspace to use in this attention head -- that is, picking out certain information from each token to be included in our weighted sum.

Therefore, $$W_V$$ answers the question: "what information are we moving"?

## Queries and keys: for each previous token, how important is the information in its value?

Next, we need to compute the weights. These depend on two additional parameter matrices, $$W_Q$$ and $$W_K$$, each of shape $$(d_\text{head}, d_\text{model})$$ (the same shape as $$W_V$$).

We want these weights to represent how much each previous token should inform our prediction of the next token. To figure this out, we extract some information from $$x^{(0)}_n$$, some other information from $$x^{(0)}_1, \dots, x^{(0)}_n$$, and compute a compatibility function between the two.

Concretely, we compute a **query** from the last token: $$q_n = W_Q x^{(0)}_n$$, as well as **keys** $$k_i = W_K x^{(0)}_i$$ for every token in the context (including $$x^{(0)}_n$$). The compatibility function is the dot product: $$q_n^\top k_i$$. For numerical stability reasons, you additionally divide by the square root of the head dimension, giving us **attention scores** $$s_i = q_n^\top k_i / \sqrt{d_\text{head}}$$.

(Why divide by $$\sqrt{d_\text{head}}$$? The short answer is: it's often helpful to keep activations in your neural network at roughly the same scale throughout, and this turns out to be the right scaling value. The semi-formal argument for this is that if the entries of $$q, k$$ are independent random variables with mean $$0$$ and variance $$1$$, then $$q^\top k$$ has mean $$0$$ and variance $$d_\text{head}$$. That means you'll commonly see much larger values! But $$q^\top k / \sqrt{d_\text{head}}$$ has mean $$0$$ and variance $$1$$, which is "on the same scale" as $$q$$ and $$k$$.)

## The weighted sum

In keeping with the idea of keeping activations on the same scale, we'd also like the output of our weighted sum to be on the same scale as the input. One way to do that is to ensure the weights sum to $$1$$, making the weighted sum a weighted *average*. 

Luckily, we already know a function that does just this: softmax! So the weights we'll use (also called the **attention pattern**) are $$[a_1, \dots, a_n] = \text{softmax}([s_1, \dots, s_n])$$.

Putting this together, we end up with a "result" vector $$r_n = \sum_i a_i v_i$$: a weighted sum of the values, as promised.

We've now answered question 2: "for each token, how important is the information it's offering?"

## Output: how do we incorporate this information into the representation of the last token?

All that’s left is to project our weighted sum back to the residual stream. We do this via one last matrix multiplication: $$o_n = W_O r_n$$. The matrix $$W_O$$ plays a similar role to $$W_V$$, but in reverse: it picks out which subspace of the residual stream the data in $$r_n$$ will be stored in.

This gets added to the orignal last-token embedding: $$x^{(1)}_n = x^{(0)}_n + o_n$$.

