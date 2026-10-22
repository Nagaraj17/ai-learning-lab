# Requirement map and starter review

Reviewed October 18, 2026, against [the implementation plan](../IMPLEMENTATION_PLAN.md) and the local instructor exercise at `topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md`.

This is a source review. It does not establish that the separate starter was trained or tested. The reference directory `C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra` is preserved.

## Evidence to build

| Requirement | Tasks | Expected evidence |
| --- | --- | --- |
| Manual Transformer with allowed PyTorch primitives, including custom LayerNorm | E09–E11 | Source walkthrough and numerical/gradient/causality checks |
| 500–1,000 synthetic pairs and explicit transformation policy | E06 | Dataset manifest, annotations and split checks |
| Tokenization, embeddings and position | L01, E07–E09 | Round-trip/unknown handling and traced shapes |
| A embedding-only, B single attention, C full model with at least two blocks | E11, E17 | Matching configurations/checkpoints and actual generated outputs |
| Five additional variants and fair component comparisons | E11, E18–E19 | Eight architecture results; seeds/budgets/parameters disclosed |
| Training, next-token prediction and autoregressive generation | L03, E12–E15 | Loss/update/restore checks; real generated continuations |
| Loss/accuracy, redaction precision/recall/F1, leakage and content preservation | E14, E17–E23 | Metrics with counts plus raw outputs and limitations |
| Controlled context, temperatures, OOD and hallucination | E20 | Token-level manipulation checks and saved outputs |
| Attention inspection and single/pair head ablations | E21–E22 | Eight single-head and two pair comparisons, actual attention rows |
| Hypotheses and careful interpretation | E16, E23 | Preregistered questions, observations and reviewed verdicts |
| Working UI for trained model selection and comparisons | E24–E27 | Three Gradio views and verified end-to-end flow |
| Beginner comments, README and shared learning notebook | E03, E05, L01–L04 | Reviewed documentation plus actual learner demonstrations |
| Cloud training and laptop demonstration | C01–C03, E27 | Actual GPU/resume/CPU portability and local demo checks |
| Every team member explains the complete token path | L04, E28–E30 | Individual walkthroughs, rehearsal and final evidence checklist |

## Starter findings to address

| Source area | Finding | Consequence |
| --- | --- | --- |
| `tiny_mitra/model.py` | Q/K/V, score calculation, head shaping and causal masking are explicit; blocks are pre-LN | Useful reference, still requires numerical/gradient/mask verification |
| `tiny_mitra/model.py` | Calls `nn.LayerNorm` in blocks and final normalization | Replace with custom feature normalization to satisfy our literal implementation commitment |
| `tiny_mitra/tokenizer.py` | Word/number tokens with `<UNK>` | New names/numbers can lose spelling; position cannot restore it; settle representation before training |
| `tiny_mitra/data.py` | Templates/entities and truncation need stronger split/coverage checks | Add disjoint validation/test design, deduplication and explicit context coverage |
| `tiny_mitra/metrics.py` | Placeholder counters do not check placement; clinical token presence does not check relations | Need aligned field/span metrics, direct leakage and relation checks |
| `tiny_mitra/generation.py` | Greedy selection below a temperature threshold, sampling above it | Separate decoding method from temperature to avoid confounding the experiment |
| Training/evaluation | Token-weighting, checkpoint provenance and resume require verification | Test these explicitly; an existing file is not verified evidence |
| Existing Gradio starter | Single-checkpoint UI is only a starting point | Implement registry selection, comparisons and inspection from real artifacts |

The existing presentation is a preparation draft. Week 5's saved numbers describe Week 5, not Tiny Mitra. No Mitra benchmark or learned head role is established by this audit.
