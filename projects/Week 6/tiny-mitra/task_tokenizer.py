"""Fixed task-word tokens plus character spelling for everything else.

This changes representation, never the expected answer. For example PATIENT is
one ID, while the unseen name Olivia is still spelled O,l,i,v,i,a. The fixed
words come from the assignment/schema, not from validation/test fitting.
"""
from tokenizer import CharacterTokenizer, CONTROL, PLACEHOLDERS

# A deliberately small, declared vocabulary extension. No person names,
# numerical identifiers or addresses are added as whole-word shortcuts.
TASK_WORDS = (
    "PATIENT", "DOCTOR", "DATE", "DOB", "MRN", "PHONE", "ADDRESS",
    "DIAGNOSIS", "BIOMARKER", "TREATMENT", "VISITED", "ON", "LIVES",
    "RECEIVED", "Dr", "NSCLC", "ASTHMA", "DIABETES", "EGFR", "ALK",
    "NONE", "DrugA", "DrugB", "DrugC", "Chemotherapy",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class TaskTokenizer(CharacterTokenizer):
    """Greedy fixed pieces; a piece can also occur inside a longer word."""
    def __init__(self):
        super().__init__()
        self.tokens += list(TASK_WORDS)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.atomic = sorted(CONTROL + PLACEHOLDERS + TASK_WORDS, key=len, reverse=True)

    def metadata(self):
        return {"type": "fixed_task_words_character_fallback_v1", "tokens": self.tokens, "unsupported": "error", "fitting": "fixed assignment vocabulary; no corpus fit", "matching": "greedy longest declared piece, including occurrences inside longer words"}
