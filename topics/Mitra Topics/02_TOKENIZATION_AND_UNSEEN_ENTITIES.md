# Topic 2 — Tokenization, Vocabulary, and Unseen Entities

## Learning goal

You should be able to trace raw text into token IDs, explain every special token, and state exactly what is lost when an unseen name becomes `<UNK>`.

## 1. The model does not read strings

```text
Raw text → token strings → vocabulary lookup → token IDs
```

```text
PATIENT John Smith DIAGNOSIS NSCLC
```

may split into:

```text
["PATIENT", "John", "Smith", "DIAGNOSIS", "NSCLC"]
```

and encode as:

```text
[37, 22, 41, 12, 31]
```

The integers have no natural ordering or similarity. ID 22 is not inherently closer to ID 23 than ID 100. The learned embedding table gives each ID a trainable vector.

## 2. How our tokenizer splits text

```python
TOKEN_PATTERN = re.compile(r"\[[A-Z_]+\]|<[A-Z_]+>|[A-Za-z_]+|\d+|[^\w\s]")
```

| Pattern part | Captures | Example |
|---|---|---|
| `\[[A-Z_]+\]` | Placeholder | `[NAME]` |
| `<[A-Z_]+>` | Control token | `<OUTPUT>` |
| `[A-Za-z_]+` | Word-like token | `DIAGNOSIS`, `DrugA` |
| `\d+` | Consecutive digits | `456789` |
| `[^\w\s]` | Individual punctuation | `,`, `.`, `-` |

Thus:

```text
DR Patel, PHONE 9876543210.
```

becomes approximately:

```text
["DR", "Patel", ",", "PHONE", "9876543210", "."]
```

`[NAME]` remains one token rather than becoming `[`, `NAME`, `]`.

## 3. Vocabulary construction

The tokenizer is fitted only on training sequences:

```python
WordTokenizer().fit([example.sequence for example in splits["train"]])
```

It counts tokens, filters by minimum frequency, sorts them, and places these first:

```text
<PAD>, <UNK>, <BOS>, <EOS>, <INPUT>, <OUTPUT>
```

Test text must not be used to fit the vocabulary. Doing so lets test information influence preprocessing and contaminates the experiment.

## 4. Encoding and decoding

Encoding uses:

```python
self.token_to_id.get(token, unk)
```

Known tokens receive their ID. Missing tokens receive the same `<UNK>` ID.

Decoding reverses the ID lookup and joins tokens with spaces. Punctuation may display as `Patel ,` rather than `Patel,`. That is a simple decoder-formatting limitation, not a Transformer failure.

## 5. What happens to an unseen name

Assume training contains:

```text
PATIENT John Smith DIAGNOSIS NSCLC
```

but not `Olivia` or `Martinez`. Test input:

```text
PATIENT Olivia Martinez DIAGNOSIS NSCLC
```

may encode conceptually as:

```text
PATIENT <UNK> <UNK> DIAGNOSIS NSCLC
```

### Information that survives

- two unknown tokens occur after `PATIENT`;
- they occur before `DIAGNOSIS`;
- their positions remain distinct;
- the surrounding sentence template is visible.

### Information that is lost

- characters inside each name;
- the distinction between these names and other unseen words;
- useful word pieces or surname endings;
- any semantic information that could have come from those forms.

The model may still redact correctly by learning a structural pattern such as:

```text
tokens after PATIENT and before DIAGNOSIS are usually a name
```

That is contextual generalization, not recognition of the actual unseen name.

## 6. The `<UNK>` collision

If different unseen spans produce the same IDs:

```text
PATIENT Olivia Martinez DIAGNOSIS NSCLC
PATIENT Severe Persistent DIAGNOSIS ASTHMA
```

both may begin:

```text
PATIENT <UNK> <UNK> DIAGNOSIS ...
```

The model cannot recover which characters were present. It must rely entirely on context. Controlled templates make that feasible, but diverse clinical prose makes it risky.

## 7. Tokenization alternatives

| Strategy | Unseen text | Sequence length | Vocabulary | Complexity |
|---|---|---:|---:|---:|
| Word | Becomes `<UNK>` | Short | Moderate | Low |
| Character | Every character represented | Long | Tiny | Medium |
| Subword | Reusable pieces | Medium | Medium | Higher |
| Hybrid | Known clinical words + fallback pieces | Medium | Medium | Higher |

### Word-level

Readable sequences and heatmaps make it a good baseline. Unknown values are its central limitation.

### Character-level

`Olivia` becomes `O l i v i a`. Unknown words disappear if all characters are covered, but sequences become much longer and the tiny model must learn words from characters.

### Subword

`Martinez` might become `Mart` + `inez`. Internal evidence is preserved with a manageable sequence length. Implementing BPE, WordPiece, or Unigram would be a separate learning exercise.

### Hybrid

Keep known medical concepts such as `BREAST_CANCER` intact while splitting unknown names and identifiers into smaller pieces.

## 8. Entity-disjoint splits

Tiny Mitra uses different pools:

```text
Training: John Smith, Mary Brown, ...
Test:     Olivia Martinez, Noah Williams, ...
```

A validation function checks that test-only names and doctors do not occur in training.

This tests:

> Can the model redact a new value inside a familiar structure?

It does not test:

> Can the model understand every new sentence structure?

## 9. Three generalization levels

### Unseen value, familiar template

```text
Train: PATIENT John Smith DIAGNOSIS NSCLC
Test:  PATIENT Olivia Martinez DIAGNOSIS ASTHMA
```

### Familiar value, unseen template

```text
Train: PATIENT John Smith DIAGNOSIS NSCLC
Test:  NSCLC was confirmed for John Smith
```

### Unseen value and unseen template

```text
Olivia Martinez presented with findings consistent with NSCLC
```

The third is hardest. These conditions should be reported separately; one test-accuracy number hides what truly generalized.

## 10. Padding and its masks

Batch tensors need a common length. Short sequences receive `<PAD>`:

```text
Token IDs:      ... <EOS> <PAD> <PAD>
Attention mask: 1 1 1 1 1 1 0 0
Loss mask:      0 0 ... output ones ... 0 0
```

- **Padding attention mask:** prevents queries from reading artificial padding keys.
- **Loss mask:** prevents padded targets and prompt targets from training the model.
- **Causal mask:** prevents every real position from reading future real positions.

These masks serve different purposes.

## 11. Experiments

1. Print token strings for a note containing a name, date, punctuation, MRN, and placeholders.
2. Compare IDs for a training name and a test-only name.
3. Count `<UNK>` tokens in train, validation, and test.
4. Test unseen names containing one, two, and three words.
5. Try a character tokenizer and compare sequence length and leakage.
6. Keep entities unseen and change the sentence template.

## 12. Common misconceptions

**“`<UNK>` means this is a name.”**  
No. It only means the token is absent from the vocabulary.

**“An unknown name makes generalization impossible.”**  
Not always. Context and positions may reveal its role.

**“Unseen names mean the entire test is unseen.”**  
No. Templates, clinical terms, and placeholder patterns may remain familiar.

**“Padding and causal masks are the same.”**  
No. Padding hides empty positions; causal masking hides future real positions.

## 13. Explain without notes

For:

```text
PATIENT Olivia Martinez PHONE 9876543210 RECEIVED DrugA FOR NSCLC
```

answer:

1. What token strings are produced?
2. Which are likely `<UNK>`?
3. What information about the name survives encoding?
4. Why can the model still possibly produce `[NAME]`?
5. What if a clinical term also becomes unknown?
6. How would character or subword tokenization change the input?

## Key takeaway

Tokenization determines what evidence reaches the Transformer. Our word-level baseline is inspectable, but unseen values collapse to `<UNK>`. Generalization claims must distinguish contextual pattern learning from processing the unseen entity text itself.
