# Tiny Mitra — final engineering handoff

## October 22: clean submission prepared

Use the project's `submission/README.md` to distinguish the hand-in from working history. The new final-study-only bundle is `artifacts/bundles/tiny-mitra-final-submission.zip` (226 files, 12,700,071 bytes; SHA256 `3c378fcc6aa17858ec9b7fd2e4b75f14018e101e5c5c6d223e02f0c9c3dbe01d`). Its fresh-extraction evidence is `artifacts/results/submission_bundle_verification.json`: 35 tests, all 11 exposed checkpoint routes, 16 final checkpoint identities plus pilot, selected-demo 11/11 authored outputs, head restoration, 12 live API frames, saved final tables and 115 local links pass. Run the clean UI with `app.py --submission`; the default historical UI remains available. The model/data/training source hash is unchanged. No training was repeated and all original studies/decks/bundles remain intact.

Learning has progressed through position, linear layers, dimensions, activations/scaling and an introduction to gradients/optimizers; attention and guided application remain next. No learner milestone is closed. Current ledger:41 gates,33 DONE/4 IN PROGRESS/4 TODO. Latest browser visual approval, rehearsal, actual presentation application and cloud checks remain open. Historical readiness/ledger counts below describe their original review date.

## Earlier handoff — October 18–19

**Learning paused at Nagar's request, October19:** Resume at PREREQUISITE_KNOWLEDGE.md **Section5: Position** (currently line67), whenever Nagar returns. Sections1–4 have been discussed: source/target, fixed vocabulary, character/special tokens, token IDs versus sequence positions, shapes and embedding lookup. Nagar explained most embedding concepts correctly; clarified that our vocabulary is declared in advance and the sequence matrix is produced by lookup, not independently learned parameters. Backprop computes gradients; the optimizer updates the shared embedding table. Feature values have no guaranteed human-readable labels. These are discussion observations, not completed hands-on/mastery gates. Keep L01 IN PROGRESS; current counts remain32DONE/4INPROGRESS/4TODO. Use the 32-feature character pilot consistently and explain the guide's illustrative64-feature examples explicitly. Next teach why token identity alone does not explicitly encode position, then learned position rows and elementwise addition with a small worked example. Preserve My Understanding for Nagar. He will run a separate practice training session himself after the walkthrough, with guidance. No automatic training or scheduled teaching wake-up; resume when he returns.

Updated October 18, 2026, Asia/Calcutta. Prepared for Nagar's October 19 login and Friday October 23 presentation (30–60 minutes). Independent authorized engineering is finished. The hourly automation `finish-tiny-mitra-implementation` is **PAUSED**, confirmed by the app tool. Do not restart training or create duplicate scheduled work. Continue when Nagar is present for the remaining human/cloud gates.

## First session

Open READY_FOR_REVIEW.md, then the project's START_HERE.md. Begin with the four-note teaching pilot and one source/target/token walkthrough. Do not ask cold diagnostic questions or mark mastery from code tests. Explain, demonstrate, clarify and then ask one teach-back question at a time. Keep My Understanding for the learner.

Project: `C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra`.

- `.venv/Scripts/python.exe` reuses system Torch; `.venv-repro/Scripts/python.exe` is genuinely isolated. Both pass34 tests with one TypedStorage warning. Isolated pip check, Gradio construction and real checkpoint generation pass. No pretrained Transformer dependency is present in the isolated environment.
- `tools/start_demo.ps1` prefers the isolated environment, falls back to .venv and detects an existing project server before starting another. Run from the project directory. The current server wrapper is PID18360, child36180, `app.py --port7860`; inspect actual command lines/metadata because PIDs can be recycled. It is localhost-only. Latest source and API checks agree.
- All training, final evaluators and finisher workers have exited. The original finisher failed on an SSE timeout at04:58. That failure/log is retained; verification resumed from the failed step and passed. `finish_queue.json` now records recovered completion, without pretending the original process succeeded. Its old temporary system-awake request was released; no persistent power setting changed.

## Frozen studies and outputs

Canonical final: `submission_experiments.py`, `artifacts/mitra_final/`. All8 primary architectures and8 repeat runs completed4,000updates from RANDOM weights on ONE corpus:700train/100validation/four50-note final partitions. Vocabulary146 fixed task pieces with character fallback; width64,FFN192,heads4 where applicable,context256,AdamWlr0.0007,batch16. Full model118802parameters. Maximum saved serialized sequence86tokens; allocated context capacity does not mean all positions were trained.

Primary comparison is ALWAYS seed42/latest.pt at4,000. best.pt is retained separately and does not replace the fixed-update selection. Four architecture repeats use7/123, varying initialization AND minibatch order on shared notes. No p-values or independent-dataset claims. Most component ablations have one seed.

Protected task-source hash: `bf6418a89068cb5d0d457573435518a39a23d17c7d074d2c87b388533cb3b173`.
Dataset hash: `c3f7a797f7610d8ef45d8977340a2705c1570b63c6af367ba355a124781bdcb7`.
Producer hash: `f2ae852fc4af39e317f5fdb5528146ead5c48299e0a5720c866a5e4facce655f`.
Tokenizer hash: `519cf5e14f6eb7ecb52ad50742b7536835f42d9b04f4637fe34c5f6fd82c357a`.
Keep model/training/data producer sources frozen. Changing them invalidates registration/resume identities; create a reviewed separate experiment if a later task needs changes. Final test results must not choose a new reported checkpoint or trigger hidden retuning.

The declared development-only rule in reports/DEMO_SELECTION_PROTOCOL.json selected **final_seed7/transformer_2**, frozen in DEMO_SELECTION.json before final outcomes. It passes11/11 authored/EOS checks, including Olivia/MRN, and two additional clinical checks whose exact sources occur in training. Primary42 remains10/11 with its PHONE failure. Seed123 also passes11/11; it was not substituted after seeing final results.

Final primary full exact counts in known/known,known/new,new/known,new/new conditions:47,5,50,5/50. Selected demo7:49,5,43,5/50. In either new-layout condition, the five exact notes are five no-identifier controls. Novel-layout redaction fails severely. Zero exact-surface leakage does not establish safe transformation; omissions and clinical substitutions remain possible.

Earlier studies remain separate histories: original flat-field character8primary/8repeat/32final; corrected instructor-character8runs; initial task-piece8runs. Different corpora, token units, budgets and selection rules must not be pooled. Optional continued/extended trials were deliberately stopped; do not claim completed budgets or resume against a changed producer.

## Executed verification and review

- Final primary/repeat identity audits,32primary raw-count partition audits and4separate demo partitions pass. All200 distinct notes are reused across models, not new independent samples each time.
- `final_test/token_metrics_audit.json`:32 teacher-forced test-loss/next-token-accuracy rows recomputed from matching weights. Correct output prefixes are supplied; output/EOS token weighting differs from free-generation exact match. Last-minibatch training loss is labelled separately from whole-validation loss.
- `artifacts/results/mechanics-tests.xml` and clean-environment-tests.xml:34passed each. `clean_environment.json`: isolated dependencies and actual UI/checkpoint route pass.
- `full_ui_api_check.json`:all28 routes plus attention, intervention/restoration, eight final/eight repeat tables, controlled context,32primary/four demo partitions and findings pass.
- `live_ui_api_check.json`:12 actual incremental Gradio frames match frozen generate.py text, probabilities and EOS;32 token rows and all9 final findings load. New live_generation.py adapter preserves benchmark inference behavior; core source stayed frozen.
- `quiet_cpu_latency.json`:35 warmed checkpoint probes, rerun after other inference/evaluation probes finished. Selected demo7 median about0.058seconds for the specified16-token output, excluding load. Different output lengths/hardware limit comparisons. No isolated training-speed claim.
- `training_exposure.json`:exact sampler replay agrees for49 saved runs; random exposures are not complete epochs.
- `reports/RESULTS.md`, FINAL_FINDINGS.md/final_hypothesis_reviews.json and three final figures:actual bounded interpretations, including no-residual/smaller-model successes and no temperature-quality effect on the eight sampled notes. Reviews are AUTHOR SELF-REVIEW, not independent sign-off.
- Final42head baseline14/14identifiers,8/8notes. L0H0disabled4/14and1/8; category supportsPATIENT7,MRN2,others1. Eight singles,two fixed pairs,matched DrugA/B context,MRN edit,attention and restoration retained. Saved head evidence is42, not demo7. No universal semantic head label.
- Both notebooks compile. All learning-notebook cells previously ran locally. Current source-only Colab ZIP contains81 project files, only code/tests/synthetic data. Actual Colab/GPU recovery/CPU portability is unexecuted.

## Presentation and deliverables

Final deck: `C:/Users/Nagar/source/repos/ai-learning-lab/presentations/mitra/output/Tiny_Mitra_Final_v7.pptx`.
SHA256: `ddb5c57d6359c4578868c6c78e6ed1d14faec184140ee5e052e1d9b134f94680`.
36slides:28core/eight appendix;21native editable tables,one editable chart/workbook,detailed notes. Every v7render was individually visually inspected. Finalizer package/layout/font/chart/import checks pass with no layout warnings. No native PowerPoint application test claimed. Earlier v5/v6 are retained history, not the delivered deck.

Presenter guide: same output directory, Mitra_Final_Presenter_Guide.md. It includes30/45/60-minute routes. Team names/roles and actual timed delivery are pending. Project reports/DEMO_SCRIPT.md gives a short real-generation/head/failure sequence.

The full demo ZIP is `projects/Week 6/tiny-mitra/artifacts/bundles/tiny-mitra-demo.zip`:546 files,67,711,558bytes. It includes unchanged final PPTX and guide copies under reports/presentation,real weights,source,raw reports and notebooks. `artifacts/results/demo_bundle_verification.json` confirms fresh extraction/hash checks/all28 actual checkpoint routes/Gradio construction using isolated Python. It is intentionally outside the ZIP to avoid circularly hashing its own verification. The final curriculum evidence/final_delivery_review.json additionally checks archive hashes against current files,protected source identity,34tests each environment,36slides/notes/native objects,local links and ledger counts.

## Browser and remaining gates

Earlier browser screenshots in project reports show real final-checkpoint generation,PHONE42failure/7success,head inspection and four-model controlled context. Latest browser recheck was blocked after a connection-refused page and the browser's data-error URL. Do not claim the new incremental/test-metric UI was visually checked. No alternate browser/port/surface was used to bypass the block. Recorded-output demo_fallback.html is structurally prepared; its file preview was also blocked, not visually approved. These visual checks need the human session.

E27:latest UI visual check. E29:timed team rehearsal,actual presentation application/projector check and optional screen recording. E30:post-rehearsal final sign-off. L01–L04:learner and every presenter's independent explanation. C03:actual Google/Drive authorization,Colab GPU/interruption/recovery/downloaded CPU comparison. INPUTS_NEEDED.md lists these. No paid compute is currently needed; local/free resources sufficed. Do not ask for passwords/private patient data.

Ledger:32DONE/3INPROGRESS/5TODO,40unique gates. E28 is prepared/reviewed presentation artifacts, separated from L04/rehearsal under Nagar's prior authorization. The project as a learning/presentation commitment is not marked complete.

Preserve unrelated edits, especially root .gitignore, all learner My Understanding sections and the separate starter at C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra. No commits/push/PR were made. No subagents were requested or spawned. Stop further autonomous engineering now; return to the teaching/rehearsal/cloud gates with Nagar.
