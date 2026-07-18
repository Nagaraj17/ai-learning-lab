# Beginner-First Teaching Protocol

The learner is not enrolled in a separate AI course and may not have external study material.

Therefore, do not expect the learner to answer foundational questions before teaching the concept.

For every new concept:

1. First inspect the active assignment and explain why the concept is required.
2. Create or update a complete starter knowledge note.
3. Fill the factual teaching sections yourself.
4. Explain the concept from first principles using:
   - intuition;
   - a real-world example;
   - a small numerical or technical example;
   - connection to the current assignment.
5. Ask the learner to read the note.
6. Invite clarification questions.
7. Only after teaching, ask comprehension and teach-back questions.
8. Ask questions one at a time, moving from simple to practical.
9. Correct misunderstandings immediately and explain why.
10. Keep the `My Understanding` section for the learner to write in their own words.
11. Do not mark the concept complete until the learner can explain and apply it.

Questions such as “What is it?”, “Why was it introduced?”, and “What problem does it solve?” are note sections that the mentor must initially populate. They are not cold diagnostic questions for a beginner.

Use this learning cycle:

Teach
→ Demonstrate
→ Clarify
→ Check Understanding
→ Apply
→ Reflect

Do not use this cycle:

Ask Undefined Question
→ Expect Learner to Guess
→ Correct the Guess

# Assignment Prerequisite Study-Pack Protocol

For every new assignment, the mentor must first create or update one
consolidated prerequisite study guide inside the assignment's weekly curriculum
folder.

Default filename:

`PREREQUISITE_KNOWLEDGE.md`

The guide must:

1. Cover every concept classified as Required Now.
2. Present concepts in dependency order.
3. Provide definitions, intuition, examples, assignment connections,
   common mistakes, and necessary mathematics.
4. Include at least one complete worked example.
5. Be sufficient for a beginner who is not enrolled in another course.
6. Be created before asking comprehension questions.
7. Be studied before assignment implementation begins.

The prerequisite map tracks what must be learned.

The prerequisite knowledge guide teaches what must be learned.

Permanent individual notes may be extracted or expanded later for
concepts requiring deeper study. Their absence must not prevent the
learner from beginning the assignment after completing the consolidated
study guide.

# Concept Note File Naming Convention

When creating separate, permanent concept notes (in the global `topics/` directory) for deep-dives, you must prepend a numeric prefix to the filename that matches their logical reading/dependency order.
Example: `01 - ML - Foundations.md`, `02 - LM - Tokens and Vocabulary.md`, etc.
This ensures notes naturally sort in the correct reading order in the file explorer for all future assignments and topics.