# Common Prompt Templates

The following templates ensure that any AI teaching assistant generates content in strict adherence to the **Beginner-First Teaching Protocol** (defined in `AGENTS.md`).

## 1. Topic Generation Prompt

Use this prompt to generate a brand new topic or fill in a missing gap in the curriculum.

```text
Please generate a new learning note for the topic: [INSERT TOPIC NAME]. 

You must strictly follow the Beginner-First Teaching Protocol defined in our AGENTS.md:
1. Start by consulting our REFERENCE_MAP.md to find the best textbook source, and extract the formal rigorous definitions.
2. Break down any mathematical formulas component-by-component in plain English.
3. Provide a vivid, contrasting analogy to build beginner intuition.
4. Include a dedicated 'Where is this used in AI?' section with concrete, real-world examples.
5. Generate and embed a visual diagram illustrating the concept.
6. Do not copy-paste paragraphs from textbooks; synthesize and explain it yourself.
```

## 2. Topic Upgrade Prompt

Use this prompt if an existing topic is too brief, too complex, or lacking visual intuition.

```text
Please upgrade the existing file: [INSERT FILE PATH]. 

You must rewrite it to strictly follow the Beginner-First Teaching Protocol defined in our AGENTS.md:
1. Ensure the Formal Definition aligns with the official textbooks in REFERENCE_MAP.md.
2. Add a 'Component-by-Component Math Breakdown' for the primary formula.
3. Add a 'Beginner Intuition' section with a contrasting, vivid analogy.
4. Add a 'Where is this used in AI?' section.
5. Generate and embed a highly educational mathematical visual diagram.
6. Preserve any existing flashcards or assignment connection sections.
```
