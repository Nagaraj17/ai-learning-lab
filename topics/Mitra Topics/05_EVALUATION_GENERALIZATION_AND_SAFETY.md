# 05 - Evaluation, Generalization, and Safety

## 1. The Problem

You trained your Tiny-GPT, and the validation loss is extremely low. You check the token accuracy, and it's 98%. You deploy it to de-identify patient records. 
The model outputs a 100-word sentence. 98 of the generated tokens match perfectly. But the 2 incorrect tokens were the model accidentally printing the patient's full real name instead of `[NAME]`. 
Token accuracy told you the model was a 98% A+ student, but in reality, it just committed a massive HIPAA privacy violation.

## 2. Why We Need Something New

Global metrics like "Loss" or "Token Accuracy" suffer from class imbalance. 95% of a medical note is safe, repetitive structural words (e.g., "The patient presented with..."). A model that just blindly copies the input will get 95% token accuracy while failing the application task completely. We need specialized metrics that separate copying from redaction.

## 3. One-Line Definition

**Targeted Evaluation** involves measuring specific behaviors like Placeholder Recall, Placeholder Precision, and Direct Identifier Leakage to prove a model is safe, rather than relying on aggregate token accuracy.

## 4. Beginner Intuition / Mental Model

Imagine hiring a censor for a top-secret government document. 
If the censor blacks out the whole page, they successfully hid the secret (100% Recall), but ruined the document (0% Precision / Over-redaction). 
If the censor changes nothing, they preserved the document (100% Precision) but leaked the secret (0% Recall / Under-redaction). 
You must measure both independently to know if the censor is doing their job.

## 5. What Came Before → What Changes Now

* **Before:** Relying on `Validation Loss` and `Token Accuracy` to declare a model "done."
* **Now:** Using targeted rule-based metrics to measure exactly how often PHI is leaked and how often medical facts are preserved.

## 6. How It Works

We define strict success criteria for every output. A good output must:
1. Remove identifiers (Measured by **Leakage Rate**).
2. Use placeholders correctly (Measured by **Placeholder Recall & Precision**).
3. Retain clinical facts (Measured by **Clinical Preservation**).
4. Avoid inventing content (**Hallucination**).

## 7. Required Mathematics 

1. **Placeholder Recall:** Correctly matched expected placeholders / Expected placeholders
2. **Placeholder Precision:** Correctly matched expected placeholders / All generated placeholders
3. **Leakage Rate:** Outputs containing ANY source identifier / Total outputs
4. **Clinical Preservation:** Preserved expected clinical markers / Expected clinical markers

## 8. Complete Worked Example

**Source Input:** `PATIENT Olivia DIAGNOSIS ASTHMA`
**Expected Target:** `PATIENT [NAME] DIAGNOSIS ASTHMA`

**Model Generation A:** `PATIENT Olivia DIAGNOSIS [NAME]`
* Leakage Rate: 1 (Olivia leaked)
* Placeholder Recall: 0 (Expected `[NAME]` where Olivia was, didn't get it)
* Placeholder Precision: 0 (Generated `[NAME]` where Asthma was, which is wrong)
* Clinical Preservation: 0 (Asthma is gone)
* **Result:** Massive failure (Under-redaction AND Over-redaction).

**Model Generation B:** `PATIENT [NAME] DIAGNOSIS ASTHMA`
* Leakage Rate: 0
* Placeholder Recall: 1
* Placeholder Precision: 1
* Clinical Preservation: 1
* **Result:** Perfect operation.

## 9. Math → Code Mapping

```python
# A simple leakage check function
def check_leakage(source_text: str, generated_text: str, identifiers_to_hide: list[str]) -> bool:
    for identifier in identifiers_to_hide:
        if identifier in generated_text:
            return True # LEAK DETECTED!
    return False
```

## 10. Experiments / What-If Questions

**What if we test the model on a name it saw heavily during training?**
*Prediction:* The model will likely perform perfectly. But this only proves **Memorization**. To prove **Generalization**, we must test the model on completely Unseen Entities (names not in the training set) and Unseen Templates (sentence structures not in the training set).

## 11. Common Misunderstandings

* **"98% accuracy means 98% safe."** -> No, error severity is unequal. A single leaked name is a critical failure.
* **"Placeholder recall proves no PHI leaked."** -> No. The model could generate `PATIENT Olivia [NAME]`. The placeholder is there, but the name still leaked! You must run direct leakage scans.
* **"Attention heatmaps explain the model."** -> Heatmaps are diagnostic evidence, not proof. Just because the `<OUTPUT>` token attended to `Olivia` doesn't mean it knew to redact it.

## 12. Limitations and Trade-Offs

Automated targeted metrics are hard to write for generative models because generation is flexible. If the model generates `DIAGNOSIS: SEVERE ASTHMA` instead of `DIAGNOSIS ASTHMA`, a strict Exact Match script will mark it as a failure, even though clinically it is correct and safe.

## 13. Where It Appears in the Current Assignment

You are expected to evaluate your Tiny Mitra models using these targeted metrics. Be prepared to show your Leakage Rate on unseen data!

## 14. Where It Appears in Modern AI Systems

Evaluation pipelines ("Evals") are currently the most important part of deploying LLMs. Companies spend millions building massive Eval suites to test models for safety, toxicity, and jailbreak vulnerabilities before release.

## 15. Connection to the Next Concept

You have now learned the entire architecture and evaluation pipeline. To tie it all together, we will trace the exact mathematical journey of a token from start to finish in **Topic 6: Complete Token Journey**.

## 16. Teach-Back and Small Application Exercise

**Exercise:** Your model generated `MRN [MRN] PHONE [DATE]`. The expected output was `MRN [MRN] PHONE [PHONE]`. 
Calculate the Placeholder Recall and Placeholder Precision. Did any PHI leak?

## 17. Quick Revision Summary

Tiny Mitra succeeds only when it removes identifiers and preserves clinical meaning. Evaluation must expose both sides, separate memorization from generalization, and rely on targeted failure scans (leakage checks) rather than global token accuracy.

## 18. My Understanding

*(Write your own intuition here. How would you explain why token accuracy is dangerous to a teammate?)*

## 19. Flashcards

**Q:** What is the difference between Memorization and Generalization?
**A:** Memorization is succeeding on training data templates and names. Generalization is succeeding on combinations and names the model has never seen before.

**Q:** What is Over-redaction?
**A:** When the model removes safe clinical information (like a diagnosis) and replaces it with a placeholder.

## 20. Sources
- AI Learning Lab - Tiny-GPT Core Concepts
