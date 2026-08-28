We started with a smaller dataset for our initial experiment, but we quickly realized it was too small which meant simpler models would always easily shine.

To fix this, we generated a new synthetic dataset inspired by the HL7 Da Vinci standard, specifically incorporating Step Therapy as one of our main focus areas. a bit complex than the supply chain events.

What we did in the workflow:
We created 12 scenario families with the following distribution:

- docs_missing_denial : 106
- pended_then_approved : 106
- direct_approval : 104
- appeal_upheld : 103
- status_check_approval : 97
- cancellation : 92
- step_therapy_denial : 91
- step_therapy_exception : 90 [VAL-ONLY]
- docs_missing_resubmit_approval : 90 [VAL-ONLY]
- appeal_overturned : 90 [TEST-ONLY]
- contraindication_exception : 90 [TEST-ONLY]
- no_pa_required : 1

| Out of these 12 scenarios, 8 were used for training, while 4 were kept completely out of the training set (2 for validation and 2 for testing, as shown above). |

_Please note that we intentionally did not shuffle all 12 scenarios together, ensuring that our test and validation sets remain strictly isolated:_

**Train set:** The data the model is actively trained on.

**Validation set:** Like a practice exam before the real test. If the model stops improving here, we use early stopping to halt training.

**Test set:** The final exam. This is data the model has never seen before by the model.

* The Split was Train: 46% | Val:27% | Test: 27%

* Mean length is 19.8. Most cases lie around 14–25 tokens, so the model must predict across histories of different lengths.

* We also see the frequency of all the token appearing in these scenarios.

* STATUS_INQUIRY is exceptionally frequent; PA_REQUEST_UPDATED is next. This imbalance helps explain the low Macro F1.

> Macro F1 - Score averages all the tokens equally, the model's terrible score on those rare underrepresented token drags the entire everage down drastically.

## The Experiment.
The way we designed this is that we would start with the most simplest model and then move up the ladder.

These hypotheses are written before seeing any results. They will not be rewritten after.

#	Hypothesis	Prediction
## Section 6 — Experimental Hypotheses

> **These hypotheses are written before seeing any results.**

| # | Hypothesis | Prediction |
|---|-----------|-----------|
| **H1** | Model A (embedding only) should handle simple local transitions | Should struggle when same current event leads to different outcomes depending on history |
| **H2** | Single-head attention should help when one earlier fact determines the next event | B > A on multi-step context cases |
| **H3** | Multi-head attention may help when several types of earlier evidence matter simultaneously | C ≥ B on complex cases |
| **H4** | Transfromer block + The FFN may improve non-linear processing of gathered context | D > C on held-out scenarios |
| **H5** | LayerNorm and residual connections may improve training stability | D-no-LN and D-no-res will show higher/more variable loss |
| **H6** | A second Transformer block may improve complex paths but may overfit a small dataset | D vs D-1 may be close or may be be better than D |
| **H7** | Increasing d_ff increases parameter count but may increase generalization gap | Generalization gap grows with d_ff |


## Why do we think so?

*Explaination*: 
-----------------------------------------------
### Model A: Embedding + PE + Linear (no context) A Single Layer NN.

```text
  Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]   ← position-aware representation
     │
  x (B, T, d_model=24)
     │
  [Linear W_head]                 ← direct vocabulary projection
     │
  Logits (B, T, vocab_size=39)
  ```


**H1** : We think that for the smaller data sets simpler neural networks work better for complex datasets we need much complex nerual networks to understand and generalize the rules.

Although it has the generalized representation of the token/word along with the positonal awareness it might not be able to learn more feature or relationships between the tokens .Additioanly it would be able to only predict the next word based on the current embedding only. It cannot use the earlier events in the sequence.

-----------------------------------------------


### Model B: Model A + Single-Head Causal Attention with a  Single NN
```text
### Model B: Model A + Single-Head Causal Attention

 Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [MultiHeadAttentionNumPy, 1 head] ← Pure attention bottleneck (NO residual, NO LayerNorm)
     │  (Q·Kᵀ / √d_k) + causal mask
     │  context = attn_weights · V
     │
  out (B, T, 24)                  ← Replaces x entirely
     │
  [Linear W_head]                 
     │
  Logits (B, T, vocab_size=39)
```
**H2**: Model B would achieve better accuracy than model A.

This model should be able to perform significantly better than the simplest Model A. While both have positional awareness, Model B introduces an attention distribution. This allows the model to intelligently decide which specific historical tokens should be given the most focus. Because it can weigh the importance of all past events in the sequence, rather than just blindly looking at the current token, it will be able to predict the next token much more accurately.

-------------------------------------------------

### Model C: Multi-Head Attention
Identical strictly to Model B's flow, but initialized with num_heads=4. Still lacks residuals and normalization.

```text
Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [MultiHeadAttentionNumPy, 4 heads] ← Pure attention bottleneck (NO residual, NO LayerNorm)
     │  concat(head_1, head_2, head_3, head_4)
     │
  out (B, T, 24)                   ← Replaces x entirely
     │
  [Linear W_head]                 
     │
  Logits (B, T, vocab_size=39)

```
 **H3** :  Multi-head attention may help when several types of earlier evidence matter simultaneously ; C ≥ B on complex cases

 We think that multi-head attention might achieve similar score to that of the single ahead attention in most of the cases but for complex cases where there are multiple categories or routes for the same event then mha might slightly perform better that that of the single head attention since it can learn relationships which might have discarded by the single head attention.

# Model D: Full Pre-LN Transformer Block

```text

### Model D: Full Pre-LN Transformer (2 Blocks)

  Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [ Block 1: TransformerBlockNumPy ]
     ├─ norm1 = LayerNorm(x)
     ├─ attn_out = MultiHeadAttentionNumPy(norm1)
     ├─ x1 = x + attn_out                          ← [Residual 1]
     ├─ norm2 = LayerNorm(x1)
     ├─ ffn_out = FeedForwardNumPy(norm2)
     └─ x2 = x1 + ffn_out                          ← [Residual 2]
     │
  [ Block 2: TransformerBlockNumPy ]
     ├─ norm1 = LayerNorm(x2)                      ← Input is output of Block 1
     ├─ attn_out = MultiHeadAttentionNumPy(norm1)
     ├─ x3 = x2 + attn_out                         ← [Residual 3]
     ├─ norm2 = LayerNorm(x3)
     ├─ ffn_out = FeedForwardNumPy(norm2)
     └─ x4 = x3 + ffn_out                          ← [Residual 4]
     │
  out (B, T, 24)
     │
  [Linear W_head]                 
     │
  Logits (B, T, vocab_size=39)

  ```

**H4** : The FFN may improve non-linear processing of gathered context | D > C on held-out scenarios. 
**Why Model D will outperform the previous models:**
While Models B and C introduce attention mechanisms, they act merely as "gatherers" of historical context. Furthermore, without residual connections, they risk overwriting the current token's identity with that history. 

Model D introduces a complete Transformer architecture that fundamentally changes how the model processes information:
1. **Non-Linear Reasoning (FFN):** While attention gathers the historical ingredients (e.g., finding both a drug category and a previous denial reason), the Feed-Forward Network acts as the "brain" that applies complex, non-linear IF/THEN logic to those ingredients.
2. **Perfect Memory (Residuals):** The residual connections ensure the model never loses the original token's meaning while processing the context.
3. **Hierarchical Processing (2 Blocks):** By stacking two blocks, the model can reason in stages—using the first block to understand the general workflow state, and the second block to refine the exact prediction based on deep Step-Therapy rules. 

Because of this, Model D will be the only model capable of fully generalizing the complex, multi-step conditions found in our held-out test scenarios.

-----------------------------------------------

## Model D-no-LN: No Layer Normalization

```text

Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [ Block 1: TransformerBlockNumPy (use_ln=False) ]
     ├─ attn_out = MultiHeadAttentionNumPy(x)      ← NO norm1 (raw input)
     ├─ x1 = x + attn_out                          ← [Residual 1]
     ├─ ffn_out = FeedForwardNumPy(x1)             ← NO norm2 (raw input)
     └─ x2 = x1 + ffn_out                          ← [Residual 2]
     │
  [ Block 2: TransformerBlockNumPy (use_ln=False) ]
     ├─ attn_out = MultiHeadAttentionNumPy(x2)     ← NO norm1
     ├─ x3 = x2 + attn_out                         ← [Residual 3]
     ├─ ffn_out = FeedForwardNumPy(x3)             ← NO norm2
     └─ x4 = x3 + ffn_out                          ← [Residual 4]
     │
  [Linear W_head]
  ```

  ## Model D-no-res: No Residual Connections

  ```text
  Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [ Block 1: TransformerBlockNumPy (use_res=False) ]
     ├─ norm1 = LayerNorm(x)
     ├─ x1 = MultiHeadAttentionNumPy(norm1)        ← NO skip connection (x is overwritten)
     ├─ norm2 = LayerNorm(x1)
     └─ x2 = FeedForwardNumPy(norm2)               ← NO skip connection (x1 is overwritten)
     │
  [ Block 2: TransformerBlockNumPy (use_res=False) ]
     ├─ norm1 = LayerNorm(x2)
     ├─ x3 = MultiHeadAttentionNumPy(norm1)        ← NO skip connection
     ├─ norm2 = LayerNorm(x3)
     └─ x4 = FeedForwardNumPy(norm2)               ← NO skip connection
     │
  [Linear W_head]
  ```
  
**H5**: LayerNorm and residual connections may improve training stability | D-no-LN and D-no-res will show higher/more variable loss

We hypothesize that removing these stabilizers will heavily degrade training performance.

D-no-LN: Without Layer Normalization, the raw mathematical scale of the embeddings will grow unchecked as they pass through the heavy matrix multiplications of the Attention and FFN layers. This will cause exploding or vanishing gradients, leading to highly unstable and erratic loss spikes during training.

D-no-res: Without residual connections, the model loses its "memory highway." The output of the Attention layer completely overwrites the original token's identity, causing severe information loss ("amnesia"). Furthermore, without this bypass, learning gradients will struggle to flow backward, causing the training loss to stall at a higher value which is the vanishing gradients.
------------------------------------------------
## Model D-1: Single Transformer Block

```text
Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [ Block 1: TransformerBlockNumPy ]
     ├─ norm1 = LayerNorm(x)
     ├─ attn_out = MultiHeadAttentionNumPy(norm1)
     ├─ x1 = x + attn_out                          ← [Residual 1]
     ├─ norm2 = LayerNorm(x1)
     ├─ ffn_out = FeedForwardNumPy(norm2)
     └─ x2 = x1 + ffn_out                          ← [Residual 2]
     │
  out (B, T, 24)
     │
  [Linear W_head]

```
**H6**: A second Transformer block may improve complex paths but may overfit a small dataset | D vs D-1 may be close or D-1 might perform better than D

While Model D features two stacked blocks to enable deep hierarchical reasoning, Model D-1 utilizes only a single block. Because our dataset is relatively small and synthetic, the massive parameter count of a two-block model (Model D) might cause it to memorize the training data (overfit) rather than learn true generalizations. A single complete block (Model D-1) is fully capable of context-gathering (Attention) and non-linear reasoning (FFN). Therefore, we anticipate D-1 will perform similarly to, or potentially even better than, Model D on the unseen test set because its simpler architecture restricts it from overfitting.

-------------------------------------------------

## Model D-no-FFN: Attention-Only Blocks

```
Tokens (B,T)
     │
  [Embedding] + [Sinusoidal PE]
     │
  x (B, T, d_model=24)
     │
  [ Block 1 & 2: TransformerBlockNumPy (use_ffn=False) ]
     ├─ norm1 = LayerNorm(input)
     ├─ attn_out = MultiHeadAttentionNumPy(norm1)
     └─ output = input + attn_out                  ← Skips norm2 and FFN entirely
     │
  [Linear W_head]
```
 This is basically learning the linear relationships with absolutely no feed forward nerwork this would perform poorly.

 ---------------------------------------------------------------
 
**H7**: Increasing d_ff increases parameter count but may increase generalization gap | Generalization gap grows with d_ff

In this model we are increasing the parameters of rhe FNN to see if the more the parameters more the accuracy and more the generalization or will the accuracy and the generalization take a hit.

We anticipate that more the parameters more the memorization, which means the model can perform exceptionally well but might fail in the test accuracy.


---------------------------------------------------------------

# Analysis of all the results:

### The Test Accuracy

| Model | Params | Test Loss | Test Acc% | Macro F1 | Gen Gap | Epochs |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: |
| **A** | 1911 | 1.8361±0.0158 | 49.8±0.4 | 0.2777 | +0.7330 | 487 |
| **B** | 4239 | 2.2420±0.0403 | 53.8±1.5 | 0.3046 | +1.2890 | 260 |
| **C** | 4239 | 2.3720±0.0996 | 53.0±1.3 | 0.2715 | +1.2674 | 193 |
| **D** | 16215 | 1.8073±0.0494 | 56.1±1.2 | 0.3445 | +0.9403 | 135 |
| **D-1** | 9063 | 1.9029±0.0686 | 55.3±0.9 | 0.3213 | +0.9912 | 125 |
| **D-no-FFN** | 6663 | 1.9597±0.1241 | 54.8±2.7 | 0.3250 | +0.9974 | 128 |
| **D-no-LN** | 16023 | 2.2482±0.0928 | 52.3±4.9 | 0.3008 | +1.1569 | 173 |
| **D-no-res** | 16215 | 1.8019±0.0340 | 55.1±1.3 | 0.3037 | +0.8308 | 202 |

The above results are interesting let us decode it.

Here are the some concepts that we learnt before we understand the results.

Test Loss: A measure of how wrong the model's predictions are on the unseen test data. Lower is better, meaning the model's predicted probabilities closely match the actual labels.

**Test Accuracy:** The percentage of correct predictions the model makes out of all the samples in the test set. It gives a quick high-level view of overall performance, though it can be misleading if classes are imbalanced.

**Macro F1:** The average of the F1 scores (which balance precision and recall) calculated independently for each class. Unlike standard accuracy, macro F1 treats all classes equally, making it a great metric for checking if the model performs well across all scenario families rather than just the majority ones.

**Generalization Gap:** The difference in performance (like accuracy or loss) between the training set and the test set. A large gap usually indicates overfitting—meaning the model memorized the training data instead of learning general patterns.
           >  $ accuracy(trainset) - accuracy(test set) $ larger the better.

**Epochs:** One complete pass of the entire training dataset through the model during training. Tracking performance across epochs helps us see how the model learns over time and when to apply early stopping


>## **Analysis of the test loss, test accuracy, macro F1, generalization gap, and epochs**

## Executive summary  and Drawing conlusions of our Hypothesis testing.

### 1. The Overall Winner: Model D
The data clearly shows that **Model D (Full Pre-LN Transformer, 2 Blocks)** is the superior architecture. It achieved the highest Test Accuracy (**56.1%**) and a significantly higher Macro F1 score (**0.3445**) than any other model. Furthermore, it converged very efficiently, requiring only 135 epochs compared to the baseline's 487.

### 2. Validating the Baselines (H1, H2, & H3)
*   **H1 & H2 Supported:** The jump from Model A (49.8% Acc, 0.2777 F1) to Model B (53.8% Acc, 0.3046 F1) is substantial. This proves that adding even a single head of causal attention allows the model to look back at the history and make smarter predictions than a simple linear projection. 
*   **H3 Nuance (B vs C):** Interestingly, Model C (Multi-Head Attention) performed slightly *worse* than Model B (53.0% Acc vs 53.8%). Because our dataset is relatively small and synthetic, the "single spotlight" of Model B was likely sufficient. Forcing the attention into 4 smaller heads (Model C) may have fragmented the 24-dimensional embedding too much without adding value, showing that MHA requires sufficient data complexity to shine. To fully understand this, we need to thoroughly investigate the attention heads by performing an ablation experiment. This will help us determine which heads are redundant, which are actively contributing, and exactly why the score dropped.

This verifies that complex architectures are well suited for complex and larger datasets, where as simpler neural network architectures best fits smaller and less complex datasets.

### 3. The Power of the FFN (H4)
*   **H4 Supported:** The massive leap from Model C (0.2715 F1) to Model D (0.3445 F1) confirms the hypothesis. Attention alone is just a router of information. When we added the Feed-Forward Network (FFN), the model gained the non-linear "brain" required to actually process that gathered context into complex Step-Therapy rules. 
*   This is further proven by **D-no-FFN**, which dropped in accuracy (54.8%) and F1 (0.3250) compared to the full Model D, confirming the FFN is doing heavy lifting.

### 4. Training Stability & Ablations (H5 & H6)
*   **H5 Supported (Stability):** Removing LayerNorm (`D-no-LN`) was catastrophic. Test accuracy plummeted to 52.3%, and it suffered the highest variance (`±4.9`). Removing residual connections (`D-no-res`) caused the model to struggle with "amnesia" and gradient flow, which is why it took significantly longer to train (202 epochs vs Model D's 135) and suffered a severe drop in Macro F1 (0.3037).

*   **H6 Supported (Depth):** Model D-1 (1 Block) performed exceptionally well (55.3% Acc, 0.3213 F1), trailing only slightly behind the 2-Block Model D. This confirms that on a dataset of this size, a single block is highly competitive and helps avoid unnecessary parameter bloat.

### 5. Generalization Gap (H7)
*   As predicted, adding complexity generally widened the gap between training and testing performance. Model A had the tightest gap (+0.7330), but as we introduced attention (Models B & C), the gap spiked to over +1.2. The normalization and residuals in Model D helped bring this gap back down slightly (+0.9403), proving their role in helping the model generalize rather than just memorize.

---------------------------------------------
## Understanding the learning curves.

1. What the curves show

    The training curves help us understand whether each model learned successfully, how quickly it learned, whether it overfitted, and whether its gradients remained stable.

2. Loss and overfitting

    Training loss measures performance on familiar training data, while validation loss measures performance on unseen validation data. When training loss continues decreasing but validation loss stops improving or increases, the model is overfitting.

3. Accuracy and loss measure different things

    Accuracy only checks whether the top prediction is correct. Cross-entropy loss also considers the model’s confidence. Therefore, accuracy may remain unchanged while validation loss increases if the model becomes more confident in its mistakes.

4. Early stopping

    The green dashed line shows when training stopped after validation loss failed to improve for a specified patience period. The best model is usually the checkpoint with the lowest validation loss, not necessarily the model from the stopping epoch or final epoch.

5. Gradient norm

    Gradient norm shows the overall strength of the weight updates. Smooth and controlled gradients generally indicate stable learning, while rapidly increasing or noisy gradients may indicate optimization difficulty. The trend matters more than the absolute value.

6. Models A, B and C

    Model A trained smoothly but reached a lower performance ceiling because it could not use earlier events. Models B and C used attention and learned the training data better, but their growing train–validation gaps showed stronger overfitting. Four-head attention did not provide a clear advantage over single-head attention in this experiment.

7. Complete Transformer models

    Models D and D-1 learned useful patterns much faster than A, B and C. Model D, with two complete Transformer blocks, achieved the strongest overall test accuracy and Macro F1, although its validation curve showed that it also began overfitting after its best epoch.

8. FFN contribution

    Model D-no-FFN showed that attention could retrieve earlier information even without the FFN. However, its weaker validation and test results indicated that the FFN helped transform contextual information into features useful for prediction.

9. LayerNorm and residual contribution

    Removing LayerNorm caused early validation deterioration, increasing gradients and high variation across seeds. Removing residual connections resulted in slow, noisy training and large gradients. Since D and D-no-res had the same parameter count, the difference came from improved information and gradient flow—not model size.


---------------------------------------------------------
## Conclusion

Does architecture really matter?

$$Observations $$


1. Why did the Baseline (Model A) do surprisingly well in certain areas?
We noticed that Model A achieved nearly 50% accuracy and had the best (tightest) generalization gap (+0.7330) of all the models.

Possible Reasons:

    1. The Nature of the Dataset (The 48% Accuracy):
    Model A is essentially a simple lookup table. It easily learns these highly frequent, deterministic A-to-B steps. Because these standard steps make up roughly half of the tokens in our average 20-token sequence, Model A easily scores around ~ 50% accuracy.
    However, it completely fails when the other half sequence, where it actually has to decided between step_therapy_denial or direct_approval.

    2. The Tight Generalization Gap (+0.7330):
    Model A has the lowest parameter count (1,911) and lacks both Attention and Feed-Forward Networks. Because its "brain" is so small and simple, it physically lacks the capacity to overfit. It cannot memorize complex, multi-step training sequences even if it tries. Therefore, the simple 1-to-1 transition rules it does learn apply almost identically to both the training set and the unseen test set, resulting in a remarkably stable generalization gap.

    3. Why it still loses overall (The Macro F1 and Epochs):
    While Model A is great at simple transitions, it completely fails at the conditional forks in the road (e.g., deciding between APPROVED or DENIED after an evaluation). Because it cannot look back at the history to find the missing documents or the drug categories, it just guesses the most statistically common outcome. This causes it to fail heavily on the rare, complex scenarios, which is why its Macro F1 score is so low (0.2777). Furthermore, because it struggles so hard to optimize without historical context, it took a grueling 487 epochs to finally stop learning, compared to Model D's swift 135 epochs.

---------------------------------------------------------

2. Does increase in parameters result in better accuracy and better generalization?
        
        Broadly, yes—but not consistently.

            From A to D:

            Parameters: 1,911 → 16,215
            Accuracy: 49.84% → 56.15%
            Improvement: 6.31 percentage points
            Macro F1: 0.278 → 0.344

            So additional capacity combined with the complete Transformer architecture helped.

            But the exceptions are extremely important.

            B versus C: same parameters, different result

            Both contain 4,239 parameters:

            B, one head: 53.81%
            C, four heads: 52.96%

            Therefore:

            Splitting attention into four heads did not improve accuracy, even though the parameter count remained identical.

            This shows the difference came from how the representation was divided and learned, not parameter count.

            D versus D-no-LN: almost the same parameters
            D: 16,215 parameters and 56.15%
            D-no-LN: 16,023 parameters and 52.26%

            Only 192 parameters were removed, but accuracy fell by approximately 3.89 points.

            That does not mean those 192 LayerNorm parameters supplied enormous prediction capacity. LayerNorm mainly made the entire network easier and more stable to train.

            Therefore:

            A small architectural component can improve how effectively all the other parameters are learned.

            D versus D-no-res: exactly the same parameters

            Residual connections do not add learned parameters here:

            D: 16,215 parameters, 56.15% accuracy, 0.344 F1
            D-no-res: 16,215 parameters, 55.06% accuracy, 0.304 F1

            The models have the same parameter count, but D performs better—especially on Macro F1.

The experiment showed that architecture affected not only final accuracy but also learning speed, stability and generalization.
Model A learned stable but limited current-token relationships. 
Models B and C learned contextual information but overfitted more strongly.
Model D learned useful contextual patterns faster and achieved the strongest overall test performance.
The ablation experiments of removing the res and LN and a transformer block showed why the complete Transformer structure mattered:
Attention retrieved information from earlier events.
The FFN transformed the contextual representation.
LayerNorm stabilized the representations and gradients.
Residual connections preserved information and improved gradient flow.
The second block provided an additional round of contextual processing.

**Our overall understanding is:**

> The complete Transformer did not perform better simply because it had more parameters. It performed better because its components worked together to learn contextual patterns efficiently. However, the training curves also showed that greater capacity made early stopping essential, since the models could continue improving on training data even after their performance on unseen data had stopped improving.
