# Delivery review — October 18, 2026

This is an **author self-review**, not independent sign-off. Engineering preparation does not establish presenter understanding or close actual-cloud checks.

| Scope | Executed evidence | Result and limit |
| --- | --- | --- |
| Allowed implementation | `model.py`; mechanics tests | Manual LayerNorm, Q/K/V, causal/padding masks, head merge, FFN, residual blocks and vocabulary head; no ready-made Transformer, pretrained model or LLM service |
| Final training | `artifacts/mitra_final/results/audit.json`, `repeats_audit.json` | All eight primary and eight additional runs complete at fixed update 4,000 on one corpus, from random weights |
| Generation quality | `reports/RESULTS.md`, `FINAL_FINDINGS.md` | Development-selected full seed 7 passes 11/11 authored notes; severe new-layout failures remain visible. Primary seed 42 and its PHONE failure are retained |
| Frozen final evidence | `final_test/audit.json`, `demo_final_test/summary.json` | 32 primary rows plus four separate demo rows over the same 200 distinct test notes; counts, EOS and identities checked |
| Test loss/accuracy | `final_test/token_metrics_audit.json` | All 32 saved teacher-forced rows recomputed; output/EOS token weighting. Correct prefixes are supplied; this is separate from free generation |
| Core checks | `artifacts/results/mechanics-tests.xml` | 34 tests pass; one TypedStorage deprecation warning |
| Fresh dependencies | `artifacts/results/clean_environment.json`, `clean-environment-tests.xml` | Isolated Python, dependency check, 34 tests, Gradio construction and actual checkpoint generation pass; no inherited Transformer library |
| UI/API | `full_ui_api_check.json`, `live_ui_api_check.json` | All 28 routes, attention/intervention/restoration and saved evidence pass. Twelve incremental frames match console text/probabilities/EOS; 32 token-metric rows and nine reviewed findings load |
| Browser | Earlier `reports/demo-*.png` screenshots | Final-checkpoint, head and source-control views were exercised earlier. Latest recheck was blocked after a connection-refused page. Updated live-token/test-metric visual check remains open; the block was not bypassed |
| Timing/exposure | `quiet_cpu_latency.json`, `training_exposure.json` | 35 warmed checkpoint latency observations and exact sampler replay for 49 runs. Latency rerun after other probes finished; no concurrent training. Training elapsed times are not isolated architecture speed benchmarks |
| Notebooks | `learning_notebook_execution.json`; notebook preparation output | Learning code cells ran locally; both notebooks compile. Colab itself and GPU-to-CPU recovery remain unexecuted |
| Presentation | `reports/presentation/` copy; source deck under repository `presentations/mitra/output/` | 36 slides: 28 core, eight appendix; 21 native editable tables and one editable chart with workbook. Every final slide rendered and visually inspected; detailed notes checked against saved evidence. No native PowerPoint application test claimed |

Corrections during review: the original verification queue stopped on an SSE connection timeout; the rerun passed. Added an actual incremental generator with parity tests, and exposed recomputed final test loss/accuracy. The first new deck export failed chart-workbook precision checks; changed the chart to exact counts rather than long fractional literals. A draft OOD example carried a target label; the final deck labels it **actual generated output**. Small training losses retain significant digits. The final deck distinguishes primary seed 42, development-selected demo seed 7, and the separate Week 5 task.

The nine final interpretations explicitly preserve unexpected results: no consistent extra-block benefit, no automatic four-head advantage, better outputs for the no-residual run on some measurements, and no sampled temperature-quality difference on the eight notes. Head roles remain tentative; no p-values or universal semantic head labels are claimed.

The downloadable ZIP is checked separately by `tools.verify_demo_bundle`: its report is outside the archive so a manifest cannot circularly hash its own final verification. Consult `artifacts/results/demo_bundle_verification.json` for the actual result and ZIP fingerprint. Do not infer successful extraction merely from this review note.

Remaining gates: latest browser visual check, local PowerPoint/projector check, timed team rehearsal and optional screen recording, every presenter tracing a prompt, and an actual user-authorized Colab session. Free local CPU resources were sufficient; no paid budget decision is currently needed. The separate starter and unrelated `.gitignore` edits remain untouched.
