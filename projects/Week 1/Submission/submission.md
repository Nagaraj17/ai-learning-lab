# Week 1 Learnings: Next Word Predictor

1. We did brush up what is supervised learning
   Any machine learning algorithm training against a labeled data is considered as supervised learning.

2. We learnt how to represent text data as numbers. 
   We used one hot encoding to represent the text data as numbers.
   2Q.We also learnt why only one hot encoding and not passig the token index directly?
     Because the one hot encoding represent the token in a way that the model can understand it, additional it understands without establishing any linear relationships with other words; otherwise we directly pass the token index the the model might think there is linear relationship between the vocabs when they are not.

3. We refreshed our Math skills on vectors, scalar, Matrix multiplication, Dot product.
   Vector a 1 D Array.
   Scalar a 0 D Array which is just a number, No dimention.
   Matrix a 2 D Array.
   use of numpys and how it automatically tansposes the Input vector and adds dimesnion automatically without us doing anything. 

4. We learnt about building blocks of neural network
   NN are bassically compose of Input layer, Hidden layer and Output layer.
   There are weights and biases associated with each layer and these wieghts and biases are the ones that are tuned to get the desired output.
   The universal equation => Y = wx+b
   where y is output, w is weights, x is input, b is bias.
  4Q How does the Input vector know where to go?
      The input vector knows where to go because of the weights and biases. The weights and biases are the ones that are tuned to get the desired output. Most important it captures which feature is important and which is not.
  4Q. What is the Shape of the Weight matrix and the bias vector?
      Weight matrix shape is (input_size, output_size) and the bias vector shape is (output_size,)
  4Q. How is the weight and the bias values intialized?
      The weight and bias values are randomly initialized, We learnt that there are many methods of random initialization 
      and the most prominent is gaussian initialization. It breaks the symmetry between neurons. (It has some connection to which activation function that we use)
  4Q. The question was why we use Y = wx+b and not something else?
        We use this equation since it is mathematically simple and easy to compute.(Matrix multiplication becomes easy and fast)
        with this comes a challenge of non linearity. Wx+b is a linear equation, which means it can only learn linear patterns.
        This is where Activation Functions help us to introduce non linearity in the model. Which helps the model to learn complex patterns.
        Some examples of Activation functions are TanH and ReLU.
  4Q. How to choose activation function?
        The choice of activation function depends on the type of problem we are trying to solve.
        if we want to go with classification problems then we use sigmoid or softmax.
        if we want to go with regression problems then we use linear activation no activation at all.
        if we want to go with hidden layers then we use TanH or ReLU.
        Somehow while discovery we learnt that ReLU fits better for the complex NN but why is something we are still Learning.
  4Q.  Why Softmax the the end over other activations?
        Softmax gives us probabilities and the sum of all probabilities is 1.
  4Q. WE also learnt about Argmax
  4Q  We learning about training and test data.

 Note!: We have run the complete forward pass and backward pass and now we have the weights and biases.
  But with still in procees to celar some concepts.

We also concluded that due to the arbitrary nature of our dataset, the Neural Network is forced to memorize the data rather than generalizing a pattern. Because of this, it does not make sense to split the data into Training and Test sets for this specific problem
  

We identified some future learnining to cover the Complete NN.
    - Gradient decent
    - Back propogation
    - Hyperparameters
    - Loss functions
    - Bias vs Variance tradeoff
    - Regularization
        


