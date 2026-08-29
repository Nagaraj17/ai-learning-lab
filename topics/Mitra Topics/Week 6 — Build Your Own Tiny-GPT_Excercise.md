Week 6 — Build Your Own Tiny-GPT
Objective
Each team will build a Tiny-GPT from scratch.
You will NOT use:
•	Hugging Face Transformer models
•	PyTorch nn.Transformer
•	Pretrained GPT models
•	OpenAI APIs
•	Existing LLM libraries
You may use basic tensor operations and neural-network primitives.
Your Tiny-GPT should include:
1.	Tokenization
2.	Token embeddings
3.	Positional encoding or positional embeddings
4.	Causal/self-attention
5.	Multi-head attention
6.	Residual connections
7.	Layer normalization
8.	Feed-forward network
9.	At least 2 Transformer decoder blocks
10.	Final vocabulary projection
11.	Softmax / next-token prediction
12.	Autoregressive generation
The purpose is not to build a production model.
The purpose is to understand:
How does a sequence of tokens become a prediction of the next token?
________________________________________


The Rule
There are four projects.
Each team chooses one.
Once a project has been selected, no other team can select the same project.
The four projects are intentionally different, but the underlying GPT architecture should remain substantially the same.
________________________________________
 
Project 1 — Tiny Covenant
Contract & Rebate Language Model
Business idea
Pharmaceutical contracts contain recurring patterns:
•	Product
•	Tier
•	Volume threshold
•	Market-share requirement
•	Rebate
•	Effective period
•	Exclusions
Your Tiny-GPT will learn the language of simplified GPO rebate contracts.
Goal
Given the beginning of a contract clause, predict and generate the remainder.
Example:
PRODUCT DrugA TIER 1 VOLUME 100 REBATE 2 PERCENT
PRODUCT DrugA TIER 2 VOLUME 200 REBATE 4 PERCENT
PRODUCT DrugA TIER 3 VOLUME 300 REBATE ?
Expected model behavior:
6 PERCENT
Another:
PRODUCT DrugB MARKET_SHARE ABOVE 60 PERCENT REBATE
Possible completion:
5 PERCENT
________________________________________
 
Sample Training Data
Generate approximately 500–1,000 synthetic clauses.
Examples:
PRODUCT DrugA TIER 1 VOLUME 100 REBATE 2 PERCENT
PRODUCT DrugA TIER 2 VOLUME 200 REBATE 4 PERCENT
PRODUCT DrugA TIER 3 VOLUME 300 REBATE 6 PERCENT

PRODUCT DrugB TIER 1 VOLUME 50 REBATE 1 PERCENT
PRODUCT DrugB TIER 2 VOLUME 100 REBATE 3 PERCENT
PRODUCT DrugB TIER 3 VOLUME 200 REBATE 5 PERCENT

PRODUCT DrugC MARKET_SHARE ABOVE 40 PERCENT REBATE 2 PERCENT
PRODUCT DrugC MARKET_SHARE ABOVE 60 PERCENT REBATE 4 PERCENT
PRODUCT DrugC MARKET_SHARE ABOVE 80 PERCENT REBATE 7 PERCENT

CONTRACT DrugD EFFECTIVE JANUARY EXPIRES DECEMBER
CONTRACT DrugE EXCLUDES MEDICARE
CONTRACT DrugF REQUIRES FORMULARY PREFERRED
You can programmatically create many variations.
________________________________________
Expected Outcome
The Tiny-GPT should learn:
•	TIER is usually followed by a number.
•	VOLUME is followed by a quantity.
•	REBATE is usually followed by a number and PERCENT.
•	Certain products have recognizable tier structures.
•	Contract clauses have predictable grammar.
Test Prompts
PRODUCT DrugA TIER 3 VOLUME 300 REBATE
PRODUCT DrugC MARKET_SHARE ABOVE 80 PERCENT
CONTRACT DrugD EFFECTIVE JANUARY
The objective is not mathematically calculating a rebate.
The model is learning contract language patterns and structure.
________________________________________
Project 2 — Tiny Codex
Regimen Sequence Language Model
Business idea
Oncology treatment regimens often appear as structured sequences:
Diagnosis → Biomarker → Line of Therapy → Regimen → Drug
Your Tiny-GPT will learn simplified clinical pathway sequences.
This is synthetic data only. It is not intended for clinical decision-making.
________________________________________
Goal
Given part of a clinical sequence, predict likely next tokens.
Example:
DISEASE NSCLC BIOMARKER EGFR POSITIVE LINE 1 REGIMEN
Expected completion:
REGIMEN_A
Another:
DISEASE NSCLC BIOMARKER KRAS POSITIVE LINE 1 REGIMEN
Expected:
REGIMEN_B
________________________________________
 
Sample Training Data
DISEASE NSCLC BIOMARKER EGFR POSITIVE LINE 1 REGIMEN REGIMEN_A
DISEASE NSCLC BIOMARKER EGFR POSITIVE LINE 2 REGIMEN REGIMEN_C

DISEASE NSCLC BIOMARKER KRAS POSITIVE LINE 1 REGIMEN REGIMEN_B
DISEASE NSCLC BIOMARKER KRAS POSITIVE LINE 2 REGIMEN REGIMEN_D

DISEASE NSCLC BIOMARKER PDL1 HIGH LINE 1 REGIMEN REGIMEN_E
DISEASE NSCLC BIOMARKER PDL1 LOW LINE 1 REGIMEN REGIMEN_F

DISEASE BREAST BIOMARKER HER2 POSITIVE LINE 1 REGIMEN REGIMEN_G
DISEASE BREAST BIOMARKER HER2 NEGATIVE LINE 1 REGIMEN REGIMEN_H

DISEASE COLON BIOMARKER KRAS WILD LINE 1 REGIMEN REGIMEN_I
DISEASE COLON BIOMARKER KRAS MUTATED LINE 1 REGIMEN REGIMEN_J
Generate variations containing:
•	Disease
•	Biomarker
•	Biomarker status
•	Line of therapy
•	Regimen
•	Drug sequence
•	Dose-cycle tokens
Approximately 500–1,000 sequences.
________________________________________
Expected Outcome
The model should learn that the meaning of:
REGIMEN
depends on all the tokens preceding it.
For example:
DISEASE NSCLC BIOMARKER EGFR POSITIVE ...
should create a different next-token distribution from:
DISEASE NSCLC BIOMARKER KRAS POSITIVE ...
This is a good test of whether attention is actually using context.
________________________________________
Test Prompt
Compare:
DISEASE NSCLC BIOMARKER EGFR POSITIVE LINE 1 REGIMEN
versus:
DISEASE NSCLC BIOMARKER EGFR POSITIVE LINE 2 REGIMEN
Ask:
Did the model understand that the same disease and biomarker can lead to different predictions because the line of therapy changed?
________________________________________
 
Project 3 — Tiny PolicyIQ
Prior Authorization Policy Language Model
Business idea
Payer policies frequently use repetitive decision structures:
IF diagnosis...
AND biomarker...
AND prior therapy...
THEN approve...
ELSE deny...
This makes them an interesting language-learning problem.
________________________________________
Goal
Train a Tiny-GPT to complete simplified payer policy statements.
Example:
PAYER Alpha DRUG DrugA REQUIRES DIAGNOSIS NSCLC AND BIOMARKER EGFR
Expected completion:
POSITIVE APPROVE
Another:
PAYER Alpha DRUG DrugA DIAGNOSIS NSCLC BIOMARKER EGFR NEGATIVE
Expected:
DENY
________________________________________
 
Sample Data
PAYER Alpha DRUG DrugA DIAGNOSIS NSCLC BIOMARKER EGFR POSITIVE APPROVE
PAYER Alpha DRUG DrugA DIAGNOSIS NSCLC BIOMARKER EGFR NEGATIVE DENY

PAYER Beta DRUG DrugB DIAGNOSIS NSCLC REQUIRES PRIOR_THERAPY DrugA
PAYER Beta DRUG DrugB PRIOR_THERAPY COMPLETED APPROVE
PAYER Beta DRUG DrugB PRIOR_THERAPY NOT_COMPLETED DENY

PAYER Gamma DRUG DrugC REQUIRES LINE 2
PAYER Gamma DRUG DrugC LINE 1 DENY
PAYER Gamma DRUG DrugC LINE 2 APPROVE

PAYER Delta DRUG DrugD REQUIRES BIOMARKER HER2 POSITIVE
PAYER Delta DRUG DrugD BIOMARKER HER2 POSITIVE APPROVE
PAYER Delta DRUG DrugD BIOMARKER HER2 NEGATIVE DENY
Generate hundreds of combinations.
Include:
•	Payer
•	Drug
•	Diagnosis
•	Biomarker
•	Step therapy
•	Line of therapy
•	Documentation requirement
•	Approve / deny
________________________________________
Expected Outcome
The model should learn policy grammar such as:
REQUIRES → CONDITION
CONDITION SATISFIED → APPROVE
CONDITION NOT SATISFIED → DENY
But there is an important challenge:
Do not simply measure whether the model memorizes individual policy examples.
Create unseen combinations in the test set.
For example, teach:
PAYER Alpha + EGFR
PAYER Beta + HER2
and then test combinations that were held out.
________________________________________
Interesting Experiment
Train with:
PAYER Alpha DRUG DrugA ...
Then replace the payer:
PAYER Beta DRUG DrugA ...
Inspect how the model changes its prediction.
Ask:
Is the model actually attending to the PAYER token?
________________________________________
 
Project 4 — Tiny Mitra
De-Identification Language Model
This one is intentionally different.
Rather than generating domain knowledge, the Tiny-GPT learns a transformation pattern:
Sensitive text → Redacted text.
________________________________________
Goal
Given a synthetic note, generate a de-identified version.
Example:
PATIENT John Smith VISITED Dr Jones ON January 12
Expected:
PATIENT [NAME] VISITED [DOCTOR] ON [DATE]
Another:
MRN 482991 PATIENT Mary Brown DIAGNOSIS NSCLC
Expected:
MRN [MRN] PATIENT [NAME] DIAGNOSIS NSCLC
________________________________________
Sample Data
PATIENT John Smith VISITED Dr Jones ON January 12
PATIENT [NAME] VISITED [DOCTOR] ON [DATE]

PATIENT Mary Brown DOB March 4 1965
PATIENT [NAME] DOB [DATE]

MRN 482991 PATIENT Robert Green
MRN [MRN] PATIENT [NAME]

PATIENT Alice White PHONE 5551234567
PATIENT [NAME] PHONE [PHONE]

PATIENT Kevin Black LIVES 412 Main Street Dallas Texas
PATIENT [NAME] LIVES [ADDRESS]

PATIENT Susan Gray DIAGNOSIS NSCLC
PATIENT [NAME] DIAGNOSIS NSCLC
Generate 500–1,000 synthetic examples with:
•	Names
•	Dates
•	MRNs
•	Telephone numbers
•	Addresses
•	Physicians
•	Clinical terms that should not be removed
________________________________________
Expected Outcome
This project has an especially interesting challenge:
The model must learn:
John Smith → [NAME]
but preserve:
NSCLC
EGFR
DrugA
Chemotherapy
So it has to distinguish:
information that identifies the patient
from
information that describes the clinical situation.
________________________________________
 
Important Test
Give it names and IDs that never occurred in training:
PATIENT Olivia Martinez MRN 991723 DIAGNOSIS NSCLC
Expected:
PATIENT [NAME] MRN [MRN] DIAGNOSIS NSCLC
If it only redacts names seen during training, the model has memorized.
If it redacts new names, it has started to learn a pattern.
________________________________________
 
Common Architecture
All four projects should use roughly the same Tiny-GPT architecture.
For example:
Vocabulary:          50–200 tokens
Embedding dimension: 32–64
Attention heads:     2–4
Transformer blocks:  2
Context length:      16–32 tokens
FFN hidden size:     64–256
Do not obsess over these exact numbers.
Keep it small enough that the model trains quickly on a laptop.
________________________________________
 
Mandatory Baselines
Every team must compare:
Model A
Embedding
→ Linear
→ Next Token
Model B
Embedding
→ Single Attention
→ Next Token
Model C
Embedding
→ Tiny-GPT
→ Next Token
Compare:
•	Training loss
•	Test loss
•	Next-token accuracy
•	Generated sequences
•	Behavior on unseen examples
The question is:
What did the Transformer actually buy you?
________________________________________
 
Mandatory Experiment — Context Matters
Every team must create two prompts where only one earlier token changes.
Example:
PAYER Alpha DRUG DrugA ...
versus:
PAYER Beta DRUG DrugA ...
or:
LINE 1
versus:
LINE 2
The expected next token should change.
Then inspect whether the Tiny-GPT changes its prediction.
This demonstrates why attention exists.
________________________________________
 
Mandatory Experiment — Temperature
Generate with:
temperature = 0.1
temperature = 0.7
temperature = 1.2
Observe what happens.
Ask:
At what point does the model become creative?
And:
At what point does creativity become nonsense?
This introduces them to an important property of real LLM inference.
________________________________________
 
Mandatory Experiment — Hallucination
Give the model something outside its training distribution.
Example:
PRODUCT DragonFruit TIER 99 REBATE
or:
DISEASE UNKNOWN BIOMARKER XYZ
Observe what it generates.
Do not fix it immediately.
Explain:
Why did the model confidently generate something even though it had never learned the answer?
This is their first direct encounter with LLM hallucination from first principles.
________________________________________
 
Mandatory Experiment — Attention Inspection
Choose one generated prediction.
Display the attention weights.
Ask:
Which previous tokens influenced this prediction?
Don't simply make a heatmap.
Explain what you believe happened.
Then add:
This is our interpretation. What experiment could prove or disprove it?
That carries forward the lesson from Week 4.
________________________________________
 
Expected Final Demonstration
Every team should be able to type a prompt such as:
PRODUCT DrugA TIER 2 VOLUME
or:
DISEASE NSCLC BIOMARKER EGFR
and watch their model generate tokens one at a time:
100
REBATE
4
PERCENT
That is the moment I want everyone to experience.
They have built:
Tokens
  ↓
Embeddings
  ↓
Position
  ↓
Masked Multi-Head Attention
  ↓
Residual + LayerNorm
  ↓
Feed Forward Network
  ↓
Transformer Block
  ↓
Transformer Block
  ↓
Vocabulary probabilities
  ↓
Sample next token
  ↓
Put it back into the model
  ↓
Generate again
They have now built a Tiny-GPT.
________________________________________
 
The Four Choices
Project	What the Tiny-GPT learns
Tiny Covenant	Contract and rebate language
Tiny Codex	Clinical/regimen sequence language
Tiny PolicyIQ	Payer-policy decision language
Tiny Mitra	PHI/de-identification transformation language
First team to claim a project gets it.
No duplicates.
________________________________________
The Question Every Team Must Answer
At the end of your presentation, answer:
What did our Tiny-GPT actually learn?
Not:
“We achieved 94% accuracy.”
Explain:
•	What patterns did it learn?
•	What did it memorize?
•	What did it generalize?
•	When did context matter?
•	When did attention matter?
•	Where did it hallucinate?
•	What broke when you changed the architecture?
•	What would need to change before this became a real SLM?
________________________________________
 
Final Challenge
You have now built a Tiny-GPT with perhaps thousands of parameters.
Modern SLMs contain billions.
But the fundamental loop is remarkably similar:
Look at the tokens behind me → predict what comes next → append it → repeat.
The scale changes dramatically.
The fundamental idea does not.

