----
# SECOND POST: QK AND OV CIRCUITS?


### The many-headed beast

* tensors
* big $$W_O$$ matrix

### Putting it together: QK and OV circuits

**[todo: right now i'm leaning toward putting this later -- maybe too much to do it now?]**

We can now write the entire attention operation at once:

**[todo: the notation above doesn't lend itself well to this. so yeah probably best to hold off until we get to more layers, that's when this is relevant anyway]**


Notice that this involves two matrices of shape `(d_model, d_model)`: \(W_{QK} = W_Q^\top W_K\) and \(W_{OV} = W_O W_V\). *Math Framework* refers to these as the “QK circuit” and “OV circuit.” 

**[ interpret these matrices ]**

This structure limits the expressivity of attention heads in a subtle way. The matrix \(W_{OV}\) determines the read/write operations for all tokens simultaneously: you can’t take some information from one token and different information from the next. Similarly, \(W_{QK}\) works the same for all positions: **[i cant finish this sentence properly]**.

This leads to some bugs in one-layer attention-only Transformers. **[todo: skip trigram bugs?]**

## MLPs in Transformers

Now that we’ve moved information between tokens, we can do some "per-token processing."

**[describe operation]**

Attention-only Transformers are powerful enough to perform some algorithmic tasks **[link]**. But “MLP-only Transformers” would be pointless: as we saw at the very start, no amount of per-token processing can improve on bigram statistics unless there’s also a way to move information between tokens.

**[some links to “key-value lookup” and 3b1b video on transformer MLPs if you want more intuition]**