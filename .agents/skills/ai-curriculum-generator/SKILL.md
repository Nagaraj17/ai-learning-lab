---
name: ai-curriculum-generator
description: >-
  Transforms a raw technical problem statement into a highly structured, 
  first-principles AI learning module. Generates deep-dive topic notes, manual 
  math exercises, prerequisite dependency maps, and a fully explained Jupyter 
  Notebook following the strict pedagogical rules of the ai-learning-lab repository.
---

# First-Principles AI Curriculum Generator

## Overview
This skill orchestrates the creation of a new weekly curriculum module for the `ai-learning-lab` repository. It takes a raw exercise text file as input and systematically generates all required directory structures, deep-dive topic notes, manual math verification exercises, and a Jupyter Notebook. It strictly enforces a first-principles teaching approach where learners must prove math manually before executing code.

## Dependencies
- None (This is a pure orchestration and reasoning skill utilizing standard file writing capabilities).

## Quick Start
Trigger this skill by asking the agent:
> "Use the ai-curriculum-generator skill to create the Week 3 module based on the provided text file."

## Workflow

### 1. Analyze Prerequisites and Ask for Clarification
- Read the raw exercise file provided by the user.
- Identify which existing `topics/` notes are required prerequisites for the new assignment.
- Identify what new concepts need to be taught.
- **CRITICAL:** If the required mathematical concepts or prerequisites are ambiguous in the text file, **STOP** and ask the user for clarification before generating any files. Do not guess the math.

### 2. Create New Permanent Topics
- Generate new Deep-Dive notes in the global `topics/` directory.
- Continue the numerical prefix convention from the existing files (e.g., if the last note was `18`, start at `19`).
- **Required Sections for every topic note:**
  - One-line definition
  - Why it exists
  - Beginner intuition
  - The Mathematical Formula (if applicable)
  - Week X assignment connection
  - Common misunderstanding
  - Teach-back question
  - My Understanding (leave space for learner)
  - Flashcards (At least two Q&A pairs tagged with `#card`)

### 3. Generate the Weekly Curriculum Structure
- Create the new directory: `weekly curriculum/Week X-{topic-name}/`.
- Inside this directory, create the following meta-files:
  - `PREREQUISITE_KNOWLEDGE.md`: A consolidated study guide explicitly linking back to required past topics and defining the new concepts.
  - `Week X Topics in Detail.md`: An ordered list mapping concepts to the deep-dive notes.
  - `REVIEW.md`: A 4-part review testing Conceptual, Code, Math, and Debugging/What-If scenarios.
  - `REFLECTION.md`: An evidence table where the learner must paste outputs from their notebook to prove completion.

### 4. Draw the Prerequisite Map
- Create `PREREQUISITE_MAP.md` inside the weekly folder.
- Use a Mermaid `graph TD` diagram.
- Color code nodes: gray for previous week concepts, blue/colored for current week concepts. Show the dependency arrows clearly.

### 5. Create Manual Exercises
- Create a `manual-exercises/` subdirectory in the weekly folder.
- Generate Markdown files containing strict mathematical walk-throughs of the week's concepts.
- Provide hardcoded initialization matrices/vectors.
- Provide a "Learner Workspace" with blank calculation tables for them to perform the math by hand.
- **Verification:** Ensure the expected outputs you write in the exercise are mathematically correct.

### 6. Build the Jupyter Notebook
- Create the notebook in `projects/Week X/Work/` (or the equivalent project directory).
- **CRITICAL:** The notebook MUST contain detailed Markdown cells explaining what is happening at every step. Do not write monolithic code blocks; break them up with explanations.
- Ensure any new third-party libraries used in the notebook are added to the root `requirements.txt`.

## Common Mistakes
- **Skipping the Math Check:** Generating manual exercises with incorrect numbers or shapes. Always double-check your matrix multiplication shapes and arithmetic.
- **Failing to Link Past Concepts:** Treating a week in isolation. The curriculum is a continuous journey; you must explicitly link new concepts back to the foundational math from previous weeks.
- **Monolithic Notebooks:** Creating a Jupyter notebook with very little markdown text and huge code cells. The notebook is a teaching tool, not just an answer key. Break code down and explain the *why*.
