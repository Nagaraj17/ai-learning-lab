# Week 5 Learnings: Building a Tiny Transformer Block from Scratch

## The question this week actually asks
Week 4 built the mechanism that decides **where to look** (multi-head attention). Week 5
asks: once attention has decided what matters, what turns that into something useful?
And, per Week 4 note on scientific restraint, we treated "does this
architecture actually help" as something to measure, not assume.

## What we built
A synthetic GPO workflow dataset (260 sequences) generated from 3 underlying families
built out of the assignment's 5 base sequences:
- **Family A (replenishment cycle):** `Order → Shipment → Receive → Restock → Inventory → Forecast → Order → ...`
- **Family B (scenario path):** `Inventory → Forecast → Scenario → Contract → Purchase → Rebate → NCR`
- **Family C (PO path):** `PO → Shipment → Invoice → Reconcile`

`Forecast` appears in both A and B with a different correct next-word each time; `Shipment`
appears in both A and C the same way — deliberate ambiguity, resolvable only from context.

We then built, from scratch in NumPy:
- **Feed-Forward Network (FFN):** `Linear(d, 4d) → ReLU → Linear(4d, d)`, applied per
  position. In plain words: attention decides what information matters; the FFN decides
  what to do with it. We expand before shrinking to give the ReLU nonlinearity more room
  to reshape the mixed representation before compressing it back down.
- **Layer Normalization:** rescales each position's vector back to a stable range after
  every sub-layer, the same instinct as subtracting `max(logits)` before softmax in Week 1.
- **Residual connections:** each sub-layer adds its output on top of its input
  (`R = X + Sublayer(X)`), giving the network an easy fallback path instead of having to
  relearn the whole transformation from scratch.
- **One full Transformer block:** `Input → MHA → +Residual → LayerNorm → FFN → +Residual → LayerNorm`
- **Two stacked blocks**, feeding into a final linear layer + softmax to predict the next
  workflow token.

We verified the full backward pass — including through both stacked blocks and the
trickiest formula, LayerNorm's gradient — against numerical gradient checks before
trusting any of it to train. All checks matched to within floating-point precision.

## A mistake we caught before trusting our own results
Our first attempt at a "held-out generalization test" used `Inventory` as an unseen
starting word for Family A. We caught, before reporting anything, that `Inventory` is
*also* Family B's own genuine starting word — so a 2-token context like
`[Inventory, Forecast]` is identical to real Family B training data. No architecture could
resolve that fairly; it wasn't testing generalization, it was an unwinnable trick question
created by how we built the test set. We fixed this by using `Restock` and `Shipment` —
words that never start any training sequence in *either* family — for the real test.

## The three-model comparison
We trained three models on identical data, embedding size, and epoch count:
- **Model A:** Embedding + Position → Linear → Prediction (no attention at all)
- **Model B:** Embedding + Position → Multi-Head Attention → Prediction (Week 4's architecture, no FFN/LayerNorm/residual)
- **Model C:** Embedding + Position → 2 stacked Transformer blocks → Prediction (this week, full)

**An early, unplanned finding:** Model B diverged to `NaN` at the same learning rate the
other two trained fine at. Model C, with LayerNorm after every sub-layer, was stable at
that same rate. We had to lower Model B's learning rate just to get it to train at all —
a direct, measured demonstration of what LayerNorm buys, not a hypothesis.

**Final results** (450 training pairs, 40 epochs, held-out generalization test):

| Model | Train Acc | Train Log-Loss | Test Acc | Test Log-Loss |
|---|---|---|---|---|
| A (linear only) | ~97% | low | **~100%** | **lowest** |
| B (bare attention) | 100% | ~0 | ~75% | high (~2.7) |
| C (full Transformer, 2 blocks) | 100% | ~0 | ~94% | low |

## Reading this honestly
- **Model B overfits.** Perfect training accuracy, but it memorized surface patterns
  rather than the underlying transition rule — generalization drops sharply, and its
  confidence in wrong answers (log-loss) is high. Nothing in its architecture regularizes
  or stabilizes it.
- **Model A does surprisingly well.** Most of this dataset's transitions are close to
  deterministic given just the last token, so a model with *no* context mechanism still
  generalizes best on raw accuracy. This is a genuinely humbling, honest result:
  architectural sophistication isn't automatically rewarded if the task doesn't need it —
  directly echoing the Week 4 mentor note that "multi-head... did not improve final
  accuracy on this dataset."
- **Model C is the most reliable, not necessarily the most accurate.** It matches Model
  A's generalization closely while being far better calibrated than Model B, and it never
  showed the training instability Model B did. The honest claim: the added components
  (FFN, LayerNorm, residuals, depth) bought **stability and calibration**, not a
  guaranteed win on raw accuracy for this particular dataset.
- **Even Model C isn't perfect.** In our spotlight test on a genuinely unseen sequence,
  the full Transformer still got one case wrong. A verified backward pass and a
  sophisticated architecture don't guarantee correct predictions on every unseen input,
  especially with only 450 training pairs and plain SGD.

## What did each component actually buy us?

**Feed-Forward Network:** not isolated separately in this experiment, but conceptually —
attention mixes information *across* positions; the FFN reshapes what a single position's
now-mixed representation means, independently per token.

**LayerNorm:** measured directly — training stability. Concrete finding, not a guess.

**Residual connections:** not isolated separately here either — a good candidate for a
follow-up experiment (train a version of Model C with residuals removed and compare).

**Depth (2 stacked blocks):** the assignment's hypothesis was that Block 1 might learn
simple associations while Block 2 refines them using longer context. **We did not verify
this directly** — doing so honestly would require inspecting each block's own attention
patterns separately, which we haven't done. Flagging this explicitly as an untested
hypothesis rather than a demonstrated finding, per the Week 4 mentor's core lesson:
state what the experiment proves, then label the rest as hypothesis until tested.

## The single clearest takeaway
The fancier architecture did not straightforwardly "win" on raw accuracy against the
simplest possible model on this dataset — but it was dramatically more stable to train and
better calibrated than the mid-complexity option. Architecture mattered most for **how
reliably** the model trained, not automatically for **how accurate** the result was.
Whether this holds on a genuinely harder task (one where the simple model *can't* just
rely on the last token) is an open question for a future week, not something we're
claiming here.
