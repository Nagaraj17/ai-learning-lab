Week 2: Embedding
______________________

Week 1 Goal: How does a neural network learn?

Although Week 1  assignment is finished we had a few topics that were pending since week 1 was about th forward pass.

We had to understand what is back propogation and why is it used and also why only back propogation algorithm are used to find the right values of weights and are there any other ways that we can find the weights?

#Moment of discovery on why do we need backgrpogation and gradient descent:
Lets take a purchase example where I am buying 3 apples where x = 3 and  y = 15 (i.e, x is no of apples and; y is the total cost of it)

So we can represnt this as linear equation
f(x) = y   where f(x) = w*x = y

w*(3) = 15 now w = 5 so we know that y = 5*x this is for one observation and one feature.

where we can say that y = 5*x here weight w = 5 
w= (1/x) * y

While when it is a one feature then it is simple to find the `w`; we simply divide the output by the input

However,When `y` is a function of multiple features (i.e, when we have apples, oranges and any other fruits) we simply cannot represet them in terms of simple scalar we would need a matrix .

like 
X = [1,2,3]
Y = [5,10,15]

And when there are multiple such observations then it becomes a matrix.
Lets consider an example where two features apples, oranges with 1000 observations and we also have 1000 outputs for each observation.
the shape for the x and y are.
x = (1000x2) [i.e, 1000 rows for observations and 2 columns for features`]
Y =(1000x1) [i.e, 1000 rows for observations and 1 column for outputs]

Previously when there was only one feature we simply divided the input or multiple (1/x) * y. While inmatrix we cannot do (1/x)*y because we cannot do a inverse of a non square matrix ( m!=n, rows and columns are different).  
We can only do a inverse of a square matrix which is 2x2, 3x3 where as our matrix is 1000X2

So we cannot directly do w = Xinverse *y

in order to make it a square matrix we need to multiple (Xtranspose * X) cause Xtranspose * X will give us square matrix like    (Xt * X) where as our original matrix is non square matrix(1/x)

X: 1000x2
Xt : 2x1000 (xtranspose is nothing but conveting rows to columns and columsn to rows)

Now multiply Xt
so the final dimension would be 
Xt * X  = (2x1000) * (1000x2) = 2x2 [square matrix ]
This squanre matrix can we inversed.

we kow that
Xw~y

lets multiply Xt to both the sides
XtXw = Xty
========================================================================
STEP 1: Starting System of Linear Equations
========================================================================
                 X     ·    w    =    y
              (1000x2)    (2x1)     (1000x1)

------------------------------------------------------------------------
Problem: X is non-square (1000x2), so X has no direct inverse!
Solution: Multiply Xᵀ on the LEFT of BOTH sides to make a square matrix.
========================================================================

========================================================================
STEP 2: Left-Multiply Xᵀ on Both Sides
========================================================================
         Xᵀ  ·  ( X   ·   w )   =   Xᵀ   ·   y
      (2x1000) (1000x2) (2x1)     (2x1000) (1000x1)
      \_______________/           \_______________/
           (2x2)                       (2x1)

Resulting Equation:
                   (XᵀX)  ·  w  =  (Xᵀy)
                   (2x2)    (2x1)   (2x1)   <-- Shapes Match! ✓
========================================================================

========================================================================
STEP 3: Left-Multiply (XᵀX)⁻¹ on Both Sides
========================================================================
   (XᵀX)⁻¹ · (XᵀX) · w   =   (XᵀX)⁻¹ · (Xᵀy)
   \_____________/
       Identity (I)

   Since I · w = w:

                   w = (XᵀX)⁻¹ · Xᵀy
                 (2x1)   (2x2)    (2x1)
                 \____________________/
                         (2x1)              <-- Final Shape Matches! ✓
========================================================================


This whole operation of Xt* inverse and again multiplying the with xty took 2^3 operations. where 2 is no of features.
and since if a 2 features requires 2^3 operations, then it would be O(n^3) if n = 100000 then it would be really expensive for caculating w with these methods and hence we use gradient decent.
Adittionaly this works only when there is linear relationships where there are non linear relationships this fails. Hence we use other methods like Backpropogation algorithm and gradient decent to find the weights.



# What is Backpropogation and gradient descent? and how they work:

Before we jump into back propogation lets take a step back.Once the neural network has provided outputs in the ouput layer and now we have applied softmax activiation to transform the output from a raw scores  which can either positive or negative and be large to small to probability where summ of all probabilities is equal to 1.

once we have the output probabilities and the prediction the next most important step is to find if our prediction is false or true.

and if false then how far was our predicted value from the true value.

This is known as calculating the loss or error.

There are many types of losses 2 main loss calculated are:
1. Mean squared error which is used for regression/numerical tasks /predictions.
2. Cross entropy loss which is use for classification tasks, like next word prediction or image classification and so on.

Cross Entropy or log loss: -y(log(predc))
Basically what it does is that when the distance is more or the loss is more it adds a huge penaly while when it distance is small the penaly is also small. 

Now, once we have the loss we need to adjust the weights in such a way that the loss is minimized.

This is where gradient decent and back propogation comes into the picture.

Backpropogation is an algorithm of ccalculating the gradients for every weight by moving backward through the network.

While Gradient descnet is responsible for updatig the weights using the gradients .

### how does it reduce it How will it know how much to increase it or reduce the weights?

To reduce the error rate it needs to know which weight contributed how much to the error. Do we need to reduce the wieght? or increase the weight? this blame of knowing which weight contributer and wether to increase the weight or decrease the weight is known as gradient.

now lets first see how is the gradient calculated given the loss

consider L is loss and W is weights

now dl/dw -> this denotes how much will the loss changes given the change in the denominator that is the weight

To calculate dl/dw we need to apply chain rule

dl/dw = dl/dh * dh/dw 

dl/dw = error contribution by its weight





Week 2 Goal: How does the neural network understand that "Receive" and "Restock" are more similar than "Receive" and "Scenario"?

Welcome to the world of embeddings.

Week 2 Focus – Learning Relationships with Embeddings
_____________________________________________________

Problem Statement
___________________

Last week, we built a very simple neural network that learned to predict the next word in a sequence. 

One limitation of that approach is that every word is treated as completely independent.
A receiving workflow naturally leads to restocking, while Scenario belongs to a completely different business function.

Your Challenge
_______________

Modify last week's neural network so that instead of using one-hot encoding, it learns an Embedding for each word.

Rather than representing each word as [0 0 1 0 0 0 ...] the network should learn something similar to

Receive   → [0.72, -0.15, 0.44, ...]
Restock   → [0.68, -0.12, 0.51, ...]
Forecast  → [-0.31, 0.82, -0.09, ...]

The numbers themselves are not important. The relationships between the vectors are.


Vocabulary:
_______________
Inventory
Order
Cabinet
Drug
Invoice
Shipment
Receive
Restock
Forecast
Scenario


Training Data: Let us use the same transitions:
________________________________________________


Order → Shipment

Shipment → Receive

Receive → Restock

Restock → Inventory

Inventory → Forecast

Forecast → Order


Week 2 Objectives
_______________________

By the end of this exercise, your team should be able to explain:

1. Why one-hot encoding does not capture similarity.

2. What an embedding actually is.

3. Where embeddings are stored inside a neural network.

4. How embeddings are learned during training.

5. Why similar words eventually end up with similar vectors.


Deliverables
________________

1. A working implementation using NumPy.
2. A visualization or table showing the learned embeddings.
3. A simple explanation that a non-AI engineer could understand.


Stretch Goal
_________________

Extend the vocabulary by adding additional words and observe whether related concepts begin moving closer together.

Additional Vocabulary:
________________________
Acknowledge
Receive
Restock
Invoice
Cabinet
Shelf
Bin


Do operational words naturally cluster together?

Do inventory-related words form their own neighborhood?


If You Don't Already Know...Learn these concepts before you begin:
____________________________________________________________________

- Embedding
- Embedding Matrix
- One-Hot Encoding
- Vector
- Similarity
- Cosine Similarity
- Dot Product
- Vocabulary
- Token
- Tokenizer

Questions to Think About: Don't just implement—think.
______________________________________________________

1. If every embedding starts as random numbers, why do similar words eventually become similar?

2. Does the neural network explicitly know that "Receive" and "Restock" are related?

3. Where is the "knowledge" actually stored?

4. If we add a brand-new word that the model has never seen, what happens?

5. Could the same idea be applied beyond language? For example:
   - Drugs
   - Practices
   - Suppliers
   - Payers
   - PO States

If yes, what business problems could embeddings help solve?

Hint:
_______
A one-hot vector is like looking up a person's employee ID. An embedding is like learning who that person is.
The employee ID identifies the person. The embedding captures what the model has learned about that person through experience.


Challenge question:
______________________
If embeddings tell us what a word is, how does the model decide which previous words matter most when predicting the next one?

