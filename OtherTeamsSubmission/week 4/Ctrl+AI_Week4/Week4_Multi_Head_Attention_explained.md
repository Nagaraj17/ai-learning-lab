# Week 4: Why One Attention Head Isn't Enough

## Where You've Been

```
Week 1 → A network learns to predict the next word from a one-hot input
Week 2 → Words become embeddings — dense vectors that capture meaning
Week 3 → Attention lets the model blend in PREVIOUS embeddings before predicting
```

Your Week 3 `AttentionModel` did this, roughly:

```python
scores  = [dot(query_embedding, prev_embedding) for prev_embedding in context]
weights = softmax(scores)
context_vector = sum(w * prev_embedding for w, prev_embedding in zip(weights, context))
final_embedding = alpha * context_vector + (1 - alpha) * query_embedding
```

One set of scores. One softmax. One blended result. That's **one attention head**,
and it has a hidden assumption baked in: *there is exactly one correct way to decide
what's relevant.*

Week 4 breaks that assumption.

---

## Part 1: The Problem — One Head Can Only Ask One Question

Take the exercise's PO lifecycle:

```
Order → Shipment → Receive → Restock → Inventory → Forecast → Invoice → Scenario
```

Current word: **Forecast**. What should the model look back at?

- **Inventory** — because forecasting depends on what's on the shelf
- **Order** / **Shipment** — because forecasting depends on what's already in the pipeline
- **Invoice** — because forecasting depends on financial commitments

All three are legitimate. They're legitimate for *different reasons* — stock levels,
supply-chain timing, and money are three different kinds of relationship between
"Forecast" and the rest of the sequence.

A single attention head produces **one** set of softmax weights over the context.
If it leans toward Inventory, it must lean away from Order and Invoice — softmax
weights sum to 1, so highlighting one thing dims the others. One head is forced to
average or pick a winner between relationships that aren't actually in competition.

**Multi-head attention's fix:** don't make one head do all the work. Give several
heads their own private "lens" on the same sequence, let each ask its own question,
and combine the answers afterward.

---

## Part 2: The Missing Piece from Week 3 — Q, K, V

Look again at Week 3's scoring line:

```python
scores = [dot(query_embedding, prev_embedding) for prev_embedding in context]
```

Notice: the *same* embedding vector is used as the query AND compared directly
against the *same* embedding vectors as context. There's no room for a head to have
its "own perspective" — everyone is looking at the raw embedding, the same way,
every time.

Real self-attention fixes this with three **learned linear projections** — three
small weight matrices that reshape the embedding into three purpose-built views:

| Projection | Question it answers | Shape |
|---|---|---|
| **Q**uery  | "What am I looking for?"        | `embedding @ W_Q` |
| **K**ey    | "What do I have to offer?"      | `embedding @ W_K` |
| **V**alue  | "What do I actually pass along if picked?" | `embedding @ W_V` |

```
attention_score(i, j) = dot(Q_i, K_j) / sqrt(head_dim)     ← scaled dot-product
weights                = softmax(scores over j)
output_i                = sum(weights_j * V_j)
```

This is exactly the shape of Week 3's `compute_attention_weights` / blending code —
dot product → softmax → weighted sum — but now Q, K, and V are each **learned**,
separately, via their own weight matrix. That's the "**Linear Projection**" concept
on your exercise's concept list. Once Q/K/V are separate, learnable matrices, giving
each head its *own* `W_Q, W_K, W_V` is a small step — and that's the whole trick.

---

## Part 3: Multi-Head Attention, Mechanically

Say `embedding_dim = 4` and you choose `num_heads = 2`. Then:

```
head_dim = embedding_dim / num_heads = 4 / 2 = 2
```

Each head gets its own `W_Q, W_K, W_V`, each projecting from 4 → 2 dimensions.
Each head independently does the full attention computation — its own scores,
its own softmax, its own weighted sum — over **the same input sequence**, just
through its own 2-dimensional lens.

```
                         embedding_dim = 4
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
        Head A (dim=2)                   Head B (dim=2)
        own W_Q, W_K, W_V                own W_Q, W_K, W_V
              │                                │
        its own attention                its own attention
        weights + output (2,)            weights + output (2,)
              │                                │
              └───────────────┬───────────────┘
                        CONCATENATE  → (4,)
                               │
                         × W_O (4×4)   ← final linear projection, mixes heads
                               │
                    multi-head attention output (4,)
```

**Concatenation** just means gluing the head outputs back together (2 + 2 = 4)
before the final mix — no math, just stacking vectors.

**W_O** (the output projection) is what lets the model *combine* whatever each
head found. Without it, you'd just have two disconnected mini-attentions sitting
side by side; W_O lets column combinations of "a bit of Head A + a bit of Head B"
become the model's actual answer — the network learns the best way to blend
opinions, rather than you hard-coding an average.

---

## Part 4: A Worked Numeric Example (by hand)

Let's build toy 4-dim embeddings for tokens near "Forecast" in the PO sequence, then
split each into two 2-dim halves — pretend Head A reads dims `[0:2]` and Head B
reads dims `[2:4]` (in a trained model this split happens through W_Q/W_K, not a
literal slice — but slicing makes the arithmetic easy to trace by hand):

```
              dims 0:2 (→ Head A)   dims 2:4 (→ Head B)
Order      =  [0.9, 0.1]            [0.1, 0.8]
Shipment   =  [0.8, 0.2]            [0.2, 0.7]
Receive    =  [0.7, 0.3]            [0.6, 0.2]
Restock    =  [0.6, 0.4]            [0.7, 0.1]
Inventory  =  [0.5, 0.9]            [0.3, 0.2]
Forecast   =  [0.4, 0.8]  (query)   [0.2, 0.9]  (query)
```

**Head A** — query = Forecast's first half `[0.4, 0.8]`. Dot with each prior token:

```
Order:      0.4*0.9 + 0.8*0.1 = 0.44
Shipment:   0.4*0.8 + 0.8*0.2 = 0.48
Receive:    0.4*0.7 + 0.8*0.3 = 0.52
Restock:    0.4*0.6 + 0.8*0.4 = 0.56
Inventory:  0.4*0.5 + 0.8*0.9 = 0.92   ← highest
```
softmax → **Order 17%, Shipment 18%, Receive 18%, Restock 19%, Inventory 28%**

**Head B** — query = Forecast's second half `[0.2, 0.9]`. Dot with each prior token:

```
Order:      0.2*0.1 + 0.9*0.8 = 0.74   ← highest
Shipment:   0.2*0.2 + 0.9*0.7 = 0.67
Receive:    0.2*0.6 + 0.9*0.2 = 0.30
Restock:    0.2*0.7 + 0.9*0.1 = 0.23
Inventory:  0.2*0.3 + 0.9*0.2 = 0.24
```
softmax → **Order 26%, Shipment 25%, Receive 17%, Restock 16%, Inventory 16%**

**Look at what happened without anyone telling it to:**
- Head A ended up most excited about **Inventory**
- Head B ended up most excited about **Order / Shipment**

That's the exact ambiguity from Part 1 — "should Forecast attend to Inventory or to
upstream Orders/Shipments?" — resolved by *not* resolving it. Head A tracks the
inventory relationship, Head B tracks the upstream-pipeline relationship, and
*both* answers survive into the output because they live in different dimensions
until concatenation. A single head, forced to produce one softmax, would have had
to blur these two patterns together or pick one.

> **Note:** these embeddings are hand-picked for the illustration, not learned. Your
> actual implementation will start from random embeddings and random `W_Q/W_K/W_V`
> per head — this is what the training loop should discover on its own, just like
> Week 3's `attention_alpha` emerged from training rather than being fixed at 0.7.

---

## Part 5: The Two "High-Level" Concepts on Your List

You don't need to implement these deeply this week, but you should know why they
exist — the exercise asks for them at a high level:

**Residual connection** — after computing the multi-head attention output, add the
*original* embedding back to it: `output = embedding + multihead_output`. Why: attention
is a *correction*, not a replacement. If a head's projections are still bad early in
training, the residual guarantees the original embedding still gets through, so the
model never gets worse than "no attention at all." It's also why gradients survive
being pushed through many layers in real transformers (not something you'll see at
2-4 heads / 1 layer, but worth knowing).

> **This turned out to matter more than expected.** When we actually implemented and
> trained this (Part 9 below), removing the residual connection capped accuracy at
> 75% no matter how long we trained — *with* it, the same model hit 100% within 7
> epochs. Not a minor stability nicety here; it was the single biggest accuracy fix
> after the learning rate itself.

**Layer normalization** — rescale the output so its values sit in a consistent range
(roughly mean 0, unit variance) before it moves on. Why: your Week 3 improved model
needed a *lower learning rate* for attention because blended values could drift to
different scales than the raw embeddings and destabilize training. LayerNorm is the
architectural fix for that same problem — keep values in a predictable range so
training doesn't need such delicate hand-tuning.

---

## Part 6: Answering the Exercise's "Questions to Think About" — Intuition First

**a. If every head learns the same thing, have we gained anything?**
No — you've paid for `num_heads` times the compute and gained nothing but redundant
copies of one pattern. This is a real failure mode (called "head collapse") and part
of why your deliverable (b/c) asks you to visualize and *compare* head patterns —
if two heads' attention maps look identical, that's a signal, not a success.

**b. Why not just make one head bigger instead?**
A single softmax over one bigger space still produces *one* weighting per position.
Making it bigger doesn't remove the constraint that weights must sum to 1 across all
candidates — it just gives that one blurry compromise more numbers to work with.
Splitting into subspaces removes the competition between Inventory-relevance and
Order-relevance entirely, because they're computed independently.

**c. Does adding more heads always help?**
No — on a vocabulary this small, you'll likely hit diminishing returns fast (maybe
even with 3-4 heads), because there just aren't that many independent relationships
in the data for extra heads to specialize in. More heads mainly help when the
underlying relationships in the data are genuinely more numerous than your head count.

**d. If one head disappears, does the model still work?**
Often yes, degraded rather than broken — this is literally studied in real
transformers (attention head pruning). It's a natural experiment for your deliverable:
zero out one head's output before the final projection and see how much accuracy drops.

**e. Can heads specialize (inventory vs. finance)?**
Yes, *if* the training data actually contains distinguishable inventory-type and
finance-type relationships for gradient descent to latch onto. Specialization isn't
guaranteed by architecture — it's discovered only if it reduces loss.

---

## Part 7: The Business Challenge — Force Specialization, or Let It Emerge?

Your instinct might be to hand-assign "Head 1 = Inventory Head, Head 2 = Financial
Head." Resist it. Forcing roles has two costs:
1. You lose the interpretability *test* — if you hard-code Head 1 to only look at
   inventory tokens, you can no longer ask "did the model discover this on its own?"
2. You cap the model's flexibility to specialization boundaries that made sense to
   you but may not be where the real structure in the data lives.

Standard practice: let heads specialize **through training**, then *inspect* them
afterward (exactly what deliverable (d) asks you to do) — look at which tokens each
trained head attends to most, and describe what it seems to have converged on. If
a head genuinely tracks something finance-shaped, you'll see it in the attention
weights without having told it to.

---

## Part 8: ForecastIQ — Combining Four Opinions

```
Head 1 (Inventory Expert):    strongly attends to Inventory
Head 2 (Supply Chain Expert): strongly attends to recent Shipments
Head 3 (Financial Expert):    strongly attends to Seasonal Forecast history
Head 4 (Demand Planner):      strongly attends to Contract Pricing
```

How does the architecture combine these four opinions? **Not** by voting or
averaging with fixed weights — by concatenating all four head outputs into one
vector, then passing that through the learned `W_O` projection (Part 3). `W_O` is
where the model learns, from data, how much to trust each expert's opinion for the
final decision — and that trust can even vary by input, because `W_O` is applied to
whatever mixture of signals the four heads produced for *this specific* sequence.

This is also the key difference from **Mixture of Experts (MoE)**, which the
exercise flags as out of scope but worth previewing: MoE *routes* each input to only
a few experts (sparse — most experts sit idle for any given input) and picks which
ones to run. Multi-head attention is **dense** — every head runs on every input,
every time, and the model learns to weight/combine them afterward rather than
choosing which ones to consult.

---

## Part 9: The Improvement Journey — From a Naive First Attempt to 100% Accuracy

This is the Week 4 version of what `Week3_Attention_Training_Improved.py` did for
Week 3: build the naive version first, watch it struggle, then fix it one piece at a
time and measure the actual effect of each fix — not assume it. Every number below
comes from a real run of `Week4_Multi_Head_Attention.py` (Part 7), reproducible by
running the file yourself.

### Step 1 — the naive first attempt

Reusing Week 3's learning rate (0.1–0.3 range) and a larger init scale (0.8), with no
scaled dot-product and no residual connection:

| | Loss | Accuracy |
|---|---|---|
| Naive (lr=0.3, init=0.8, no scaling, no residual) | **NaN — diverged** | 12.5% (= random guessing on 8 words) |

The loss overflows to NaN. The model doesn't learn anything at all — it's stuck at
the random-guess floor.

### Step 2 — which single fix actually stops the divergence?

Tempting to assume it's the missing scaled dot-product or the missing residual
connection, since those are the "textbook" attention fixes. Testing each candidate
**alone**, with everything else still naive, says otherwise:

| Fix applied (alone, lr still 0.3) | Result |
|---|---|
| Sane init scale (0.8 → 0.2) only | still diverges (NaN) |
| Scaled dot-product (÷√head_dim) only | still diverges (NaN) |
| Residual connection only | still diverges (NaN) |
| **Lower learning rate (0.3 → 0.05) only** | **stable — loss 0.386, accuracy 75%** |

Only the learning rate was actually causing the divergence. With 4 heads × 3
projections (`W_Q, W_K, W_V`) plus an output projection `W_O` all updating jointly,
there are far more interacting weight matrices than Week 3's single embedding table —
the same learning rate that was fine for Week 1–2 and workable for Week 3's one
attention head was too aggressive here. This is the exact same lesson Week 3's
improved version already learned once (its 0.1 → 0.01 fix for attention) — multi-head
attention just needed it applied again, harder, because it has more moving parts.

### Step 3 — learning rate fixed. What do scaling and residual actually add?

| Config (lr=0.05 for all) | Final accuracy | Final loss | First hits 100% |
|---|---|---|---|
| A. neither scaling nor residual | 75.0% | 0.386 | never (100 epochs) |
| B. + residual connection | **100.0%** | 0.0050 | **epoch 7** |
| C. + scaled dot-product only | 75.0% | 0.387 | never (100 epochs) |
| D. + both | 100.0% | 0.0049 | epoch 7 |
| E. + sane init too (final config) | 100.0% | 0.0089 | epoch 16 |

The **residual connection** is what takes accuracy from a 75% ceiling to 100%, and
fast. The **scaled dot-product made no measurable difference on its own** (row C is
identical to row A) — because `head_dim` here is tiny (2), so raw dot products never
grew large enough to need rescaling. It's still worth keeping in the implementation
(a real transformer's `head_dim` of 64+ absolutely needs it), but on this dataset it
wasn't the fix that mattered.

One more honest surprise: the larger init scale (row D, 0.8) reached 100% *faster*
(epoch 7) than the smaller "safer" init (row E, 0.2, epoch 16) once the other fixes
were in place. "Smaller init is always safer" isn't a universal rule — it interacts
with everything else in the configuration.

### Step 4 — more epochs, once the task is already solved

| Epochs | Loss | Accuracy |
|---|---|---|
| 100 | 0.0089 | 100.0% |
| 300 | 0.0013 | 100.0% |

Same accuracy either way — extra epochs just shrink the loss further. Same
diminishing-returns pattern Week 3 saw once a small, largely deterministic task is
already solved.

### Summary, ranked by actual measured impact

1. **Learning rate (0.3 → 0.05)** — fixed catastrophic NaN divergence. The critical
   fix; nothing else mattered until this was in place.
2. **Residual connection** — took accuracy from a 75% ceiling to 100%, converging by
   epoch 7.
3. **Scaled dot-product (÷√head_dim)** — textbook-correct, cheap to keep, but made no
   measurable difference at this toy `head_dim=2`.
4. **Sane init scale (0.8 → 0.2)** — not required for correctness here; the larger
   init even converged faster in this run. Worth keeping anyway for robustness on
   less trivial data.
5. **More epochs (100 → 300)** — diminishing returns once accuracy is already 100%.

---

## What We Built

Implementation lives in [Week4_Multi_Head_Attention.py](Week4_Multi_Head_Attention.py),
following the same incremental path outlined for every concept above:
1. Real `W_Q`, `W_K`, `W_V` projections (per head) replacing Week 3's "dot product on
   raw embeddings" shortcut, validated at `num_heads=1` against Week 3's accuracy.
2. Split into `num_heads` independent projections of `head_dim = embedding_dim / num_heads`,
   concatenated and mixed through `W_O`, with a residual connection.
3. Trained on the exercise's longer sequences (`Order → Shipment → Receive → Restock → Inventory`, etc.).
4. Per-head attention visualized and compared — see
   [Week4_Multi_Head_Attention.html](Week4_Multi_Head_Attention.html) for the heatmaps,
   the ambiguous-"Forecast" test, and the ablation results.
5. The "zero out one head" and "force two heads identical at init" experiments from
   Part 6, answering questions (a) and (d) with real numbers instead of just reasoning
   about them.
6. The naive-to-fixed improvement journey above (Part 9), run and verified end to end.
