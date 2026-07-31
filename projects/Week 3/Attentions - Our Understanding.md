# Week 3 Submission on attentions.

# Before we get into attentions.

We corrected some nuances and understanding of the embedding layer itself. We misunderstood a few things previously which we corrected during our learning.

So we corrected ourselves in the below passage.

The model first tokenizes text into token IDs.
The embedding layer contains a full embedding matrix E (vocabsize,dim), which is randomly initialized at the start of training.
During the forward pass, each token ID is used to look up its row in E based on the words in the input sentence.
The selected rows are stacked to form the sequence matrix X which is nothing but the stack of embedded vectors for the words in that particular sentence, which becomes the input to the network.
During backpropagation, the embedding values in E are updated so that the model gradually learns useful representations.

Initially we thought that the output of the first layer is embedding matrix which is wrong . It is a learned parameters this embedding matrix (E) is not the output of the first layer at all it was created by using random values initially and then the values gets updated during back prop.

> Embedding Matrix (E) is nothing but the stack of embedded vectors of the entire library which is first initialized randomly.

We can look it like this it is the library of input matrix where each embedded vector for that word is book while the token id is the index/location no. of where that book is kept in the embedding matrix.

And When during training based on the sentence the sequence matrix is created picking the embedded vector from embedded matrix, while in back propogation the books are updated based on the loss.

So, it is the embedding layer.


E starts random
     ↓
many forward passes
     ↓
loss + backpropagation
     ↓
embedding rows repeatedly updated
     ↓
E becomes meaningful

One most important point that we understood here is that the embedding matrix E is not universal. It is the learned representation based on the patterns and the context present in the sentences that it is trained on which becomes useful.
Like ex:

River  bank
Bank account
Bank loan
Bank of the river

The embedding for the word bank is different in the above 4 sentences. based on the context it is learned.

Bank embedding
     ↓
learned general representation
based on all the ways "bank"
appeared during training

---X End of the correction for Week 2 X-------

Now Let's come to attentions.

# Week 3

### Attentions:

> Why do we need Attentions when we have the learned representation of the word/token?

The Embedding tells us the general representation of the word, But not the meaning in a particular sentence.

Although it has learned based on the company it kept but doesn't give meaning of the word that it is used currently in the given sentence.

Let's continue with the same example of the Bank
River  bank
Bank account
Bank loan
Bank of the river

Its learned embedding may capture that bank is related to things like:
money
loan
river
shore 
account


Bank has the same base embedding for all the above sentences, But the context of the word bank changes based on the sentence it is in.

River Bank -> context related to water, shore etc.
Bank account -> context related to money, account, etc.
Bank loan -> context related to money, loan, etc.
Bank of the river -> context related to water, shore, etc.

**This is where we need to attentions.**
Although an embedding captures a token’s general learned meaning from training, it is still a context-independent base representation at lookup time. It does not yet reflect the surrounding tokens of the current sentence. 
Here is where and why we need Attention. **Attention adds that current context**.

> **PROBLEM** While Embeddings tells us te general representation of the word based on the company it kept during the training.It Doesn't provide the semantic meaning of the word in the given context window.
It doesn't know if we are referring to financial related bank or a river bank.
> **SOLUTION** Attention tells us the meaning of the words based on current sentence/context it is currently being used in.

Attention looks at the company the token is keeping right now. what cotext it is been used

## How does Attention mechanism work or know the current context of the word?
For the given sentence "The **bank** river overflowed".






