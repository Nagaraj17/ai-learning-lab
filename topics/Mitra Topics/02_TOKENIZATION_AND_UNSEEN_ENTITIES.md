# 02 - Tokenization and Unseen Entities

## 1. The Problem

Machine learning models only understand numbers, not text. If our training data has the string `PATIENT John Smith`, we need a way to turn it into an array of integers like `[45, 102, 19]`. 
But what happens when the model is tested on `PATIENT Olivia Martinez`? If the names `Olivia` and `Martinez` were never seen during training, the tokenizer won't have an integer ID for them. The model will crash or fail because it doesn't know how to represent them!

## 2. Why We Need Something New

We need a standardized vocabulary and a safe fallback mechanism so the model can process entirely new, unseen words without crashing, while still understanding that these unknown words represent entities like names or dates.

## 3. One-Line Definition

**Tokenization** is the process of splitting text into manageable pieces (tokens) and mapping them to a fixed dictionary of integer IDs, using a special `<UNK>` (Unknown) token as a fallback for any piece of text not in the dictionary.

## 4. Beginner Intuition / Mental Model

Imagine a restaurant menu that only has 80 dishes, each numbered 1 through 80. If a customer orders "Steak", you write down `45`. If they order "Salad", you write down `12`. 
What happens if someone asks for "Sushi", which isn't on the menu? Instead of throwing them out, you write down a special code `99` which means "Off-Menu Special". The chef (the model) treats every `99` as a generic unknown dish. It doesn't know if it's sushi or tacos, but it knows it's an off-menu item.

## 5. What Came Before → What Changes Now

* **Before:** Throwing an error when encountering an unseen string in test data.
* **Now:** Gracefully falling back to the `<UNK>` token, ensuring the sequence length is preserved and the model can still process the sentence.

## 6. How It Works

We build a dictionary (vocabulary) exclusively from the training dataset. 
During tokenization (encoding):
1. The text is split into words using a regular expression.
2. We look up each word in the dictionary.
3. If it exists, we return its integer ID.
4. If it doesn't exist, we return the ID for `<UNK>`.

When decoding back to text, `<UNK>` IDs are printed as `[UNK]`, so the original unknown word is permanently lost to the model.

## 7. Required Mathematics 

There isn't much math in tokenization! It's a hash map (dictionary) lookup:
$ID = Vocab\_Dict[token] \text{ if } token \in Vocab\_Dict \text{ else } UNK\_ID$

## 8. Complete Worked Example

Suppose our fitted vocabulary is:
`{"<BOS>": 0, "<UNK>": 1, "PATIENT": 2, "John": 3, "DIAGNOSIS": 4, "NSCLC": 5}`

**Training sentence:** `PATIENT John DIAGNOSIS NSCLC`
**Token IDs:** `[2, 3, 4, 5]`

**Test sentence:** `PATIENT Olivia DIAGNOSIS ASTHMA`
* `PATIENT` -> `2`
* `Olivia` (not in vocab) -> `<UNK>` -> `1`
* `DIAGNOSIS` -> `4`
* `ASTHMA` (not in vocab) -> `<UNK>` -> `1`
**Token IDs:** `[2, 1, 4, 1]`

## 9. Math → Code Mapping

```python
class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        tokens = self.split(text)
        # Look up the ID, defaulting to self.unk_id if the token is missing
        return [self.vocab.get(token, self.unk_id) for token in tokens]
```

## 10. Experiments / What-If Questions

**What if two completely different unseen names appear next to each other?**
*Prediction:* E.g. `Olivia Martinez`. They both become `<UNK>`. The model sees `[1, 1]`. Initially, their token embeddings are identical. However, the *Positional Embeddings* added to them will be different (one is position 4, one is position 5), allowing the model to know they are two distinct tokens in a sequence, not one giant unknown blob.

## 11. Common Misunderstandings

* **"The model can figure out the word behind `<UNK>`."** -> No. Once the word is converted to `<UNK>`, the actual string `Olivia` is completely erased. The model only sees ID `1`.
* **"We should fit the tokenizer on the test data too."** -> ABSOLUTELY NOT. This is data leakage. Your model will appear to perform perfectly because it never encounters unseen data, but it will fail miserably in the real world.

## 12. Limitations and Trade-Offs

Using a simple word tokenizer with an `<UNK>` token destroys information. Modern LLMs (like GPT-4) use subword tokenization (like Byte-Pair Encoding or BPE). In BPE, if `Martinez` is unseen, it might be split into `Mart` + `in` + `ez`, which are common subwords. This eliminates the need for `<UNK>` tokens entirely! We use word tokenization here for simplicity.

## 13. Where It Appears in the Current Assignment

You will use the `WordTokenizer` class. You must ensure you only call `tokenizer.fit()` on your training data, and only call `tokenizer.encode()` on your test data.

## 14. Where It Appears in Modern AI Systems

Every LLM has a tokenizer. However, as mentioned above, modern systems use subword tokenization (BPE or SentencePiece) to handle unseen words efficiently without losing the actual characters.

## 15. Connection to the Next Concept

Now that we've turned our words into integer IDs, how do we give those arbitrary integers actual meaning? We do this by mapping them to dense vectors. We'll explore this and the core processing block in **Topic 3: Causal Decoder Block**.

## 16. Teach-Back and Small Application Exercise

**Exercise:** If your vocabulary is `{"I": 0, "like": 1, "cats": 2, "<UNK>": 3}`. What is the encoded sequence for the sentence `I like blue dogs`? 

## 17. Quick Revision Summary

Tokenization converts text strings into integer IDs using a fixed vocabulary dictionary. To prevent crashes on unseen words, a special `<UNK>` token ID is used as a fallback.

## 18. My Understanding

*(Write your own intuition here. How would you explain OOV to a teammate?)*

## 19. Flashcards

**Q:** Why must the tokenizer only be fitted on the training data?
**A:** To simulate real-world conditions where the model encounters words it has never seen before. Fitting on test data is data leakage.

**Q:** How does the model distinguish between two `<UNK>` tokens next to each other?
**A:** By adding positional embeddings to them, giving them distinct spatial locations in the sequence.

## 20. Sources
- AI Learning Lab - Tiny-GPT Core Concepts
