# Manual Exercise 2: Concatenation and Output Projection

> Objective: compute the final combination step by hand so `Concat(... )` and
> `W_O` stop feeling abstract.

## Given

Assume two head outputs for the same two-token sequence:

`head_1 = [[1.0, 0.5], [0.2, 0.8]]`

`head_2 = [[0.3, 0.7], [0.9, 0.1]]`

## Learner Workspace

### Step 1: Concatenate

Concatenate the two heads side by side along the feature axis.

`concat = [ head_1 | head_2 ]`

Write the result:

`concat =`

### Step 2: Check the shape

What is the shape of `concat`?

Answer:

### Step 3: Apply the output projection

Let

`W_O = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]`

Compute:

`output = concat W_O`

Write the result:

`output =`

### Step 4: Change `W_O`

Now let

`W_O = [[1,0,0,0], [0,1,0,0], [1,0,1,0], [0,1,0,1]]`

Compute the first row of `output`.

First-row calculation:

- column 1:
- column 2:
- column 3:
- column 4:

## Solution Key

### Step 1

`concat = [[1.0, 0.5, 0.3, 0.7], [0.2, 0.8, 0.9, 0.1]]`

### Step 2

`concat` has shape `(2, 4)`.

### Step 3

With identity `W_O`, the output is unchanged:

`output = [[1.0, 0.5, 0.3, 0.7], [0.2, 0.8, 0.9, 0.1]]`

### Step 4

First row:

- column 1 = `1.0*1 + 0.5*0 + 0.3*1 + 0.7*0 = 1.3`
- column 2 = `1.0*0 + 0.5*1 + 0.3*0 + 0.7*1 = 1.2`
- column 3 = `1.0*0 + 0.5*0 + 0.3*1 + 0.7*0 = 0.3`
- column 4 = `1.0*0 + 0.5*0 + 0.3*0 + 0.7*1 = 0.7`

So the first row becomes:

`[1.3, 1.2, 0.3, 0.7]`

## Why This Exercise Matters

This exercise makes one important point visible:

- concatenation preserves the separate head outputs
- `W_O` is where learned recombination begins
