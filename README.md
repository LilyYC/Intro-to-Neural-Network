# Machine Learning: Neural Networks
This is a repo of course project and an extension of course material covered by CSC321-18Fall (now CSC421+CSC411)

Keywords: Machine Learning Theory, predictive algorithms: bagging and gradient boosting methods, NLP, GAN, Graphical Models, Bayesian methods
Computing Techniques: Back-propagation | Automatic Differentiation

Reference: http://www.cs.toronto.edu/~rgrosse/

# Selected Conceptual Topics

## 1. Types of Learning models in ML
### A. Supervised learning:  predict with a given target
     - to predict a scalar valued target, i.e. when target variable is continuous -> Linear Regression, Polynomial Regression, Quadratic Regression.
     - to classify the result, i.e. when the target variable is categorial -> Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Gradient Boosting, ADA Boosting, Bagging, Random Forest
     * tasks: Classification/Regression/Estimation

  - core idea behind Supervised Learning Program:
    - training set (contain part of both X and Y, need to concatenate explanatory dataset and target dataset)
    - testing set / validation set
    * consideration in the split of dataset: generalization problem

  ## B. Unsupervised learning: predict with unspecified target
	Train the machine without any proper guidance, or when it’s hard to specify a target.
	The model learns through observations and deduced structures in the data.
	e.g. PCA, Factor Analysis, Singular Value Decomposition, K-means
      * tasks: clustering, prediction	

  ## C. Reinforcement Learning
	Also, no target is given, but there’s a reward/penalty mechanism. 
	It involves an agent to interact with an environment to create an action, and get reward if it behaves nicely (More code on this in the theoretical physiology course)
	e.g. Q-Learning, R-Learning
      * tasks: Decision-making

Difference between ML and Deep Learning:
- ML involves algorithms that learn from patterns of data then apply it to decision making - relies on structured data
- Deep Learning is able to learn through processing data, and is similar to the human brain where it identifies something, analyze it, and makes a decision - relies on layers of artificial neural networks

Variable Selection Technique:
- run Linear Regression and select with p-value
- Lasso Regression
- Random Forest and plot variable chart
- Top features can be selected based on information gain criteria 

## 2. Neural Networks: Multilayer perceptrons
## Conceptual Review:
In 1957, the idea of neural networks was first described as a perceptron.
Each neuron in a neural network can be think of as a building block, and is fundamentally a mathematical function. 
By definition, each neuron computes a weighted sum of inputs fed to it, where the weight assigned to each input decides the impact it has on the output. 
In order for neural networks to solve non-linear problem, we apply an activation function to the weighted sum computed by the neuron. 

The whole idea about machine learning is developed upon experiments, to test the power of a model, the idea is to split dataset into two, one training set, the other, testing set. We expect the weights for contributing input to increase, while others are decreased. By designing this procedure, we specify a learning rate, and use loss function (another design issue) to control model's learning behavior.

Another fundamental idea is differential calculus, as we are trying to explain how a small change in each input affects the model's output, in other words, we need to find out the derivative. As the layer gets deeper, computation power is a great concern. Hence, the concept of back-propagation is a bliss for computer scientists.

## 3. ML models [Problems related to Supervised Learning]
  - Linear Regression: 
      - Can be formulated as an optimization problem, 
      - can be solved
                - directly
                - gradient descent [algorithm]
      - Gradient Descent vs. Stochastic Gradient Descent

* Speed Booster Techniques / Thought Process:
  - Vectorization
  - Feature maps and polynomial regression
  - Generalization: overfitting, under-fitting, and validation.

Linear Classification
- Binary linear classification. 
- Visualizing linear classifiers. The perceptron algorithm. Limits of linear classifiers.

Learning a Classifier: Comparison of loss functions for binary classification. Cross-entropy loss, logistic activation function, and logistic regression. Hinge loss. Multiway classification. Convex loss functions. Gradient checking. (Note: this is really a lecture-and-a-half, and will run into what's scheduled as Lecture 5.)


Distributed Representations: 
  - Language modeling [NLP Natural Language Processing]
  - n-gram models (a localist representation), 
  - neural language models (a distributed representation)
  - skip-grams (another distributed representation).

Optimization
  - How to use the gradients computed by backprop
  - Features of optimization landscapes: local optima, saddle points, plateaux, ravines. 
  - Stochastic gradient descent and momentum.

Generalization
    Bias/variance decomposition 
    data augmentation
    limiting capacity
    early stopping
    weight decay 
    ensembles
    stochastic regularization
    hyperparameter tuning.

Automatic Differentiation
    How to implement an automatic differentiation system. 
    Based on Autodidact, a pedagogical implementation of Autograd

Convolutional Networks
    Convolution operation. 
    Convolution layers and pooling layers. 
    Equi-variance and invariance. 
    Back-prop rules for conv nets.

Image Classification
    Conv net architectures applied to handwritten digit and object classification. 
    Measuring the size of a conv net.

Optimizing the Input
    Interesting things you can do with gradient descent on the inputs: 
     - conv net visualizations, adversarial inputs, Deep Dream.

## Recurrent Neural Nets
  - Back-prop through time. Applying RNNs to language modeling and machine translation.
Learning Long-Term Dependencies
  - Why RNN gradients explode and vanish
    - the mechanics of back-prop
    - conceptually in terms of the function the RNN computes 
  - Ways to deal with it: gradient clipping, input reversal, LSTM

## ResNets and Attention
    Deep Residual Networks. 
    Attention-based models for machine translation and caption generation.

## Bayesian Methods
    Learning Probabilistic Models
    Maximum likelihood estimation. 
    basics of Bayesian parameter estimation and maximum a-posteriori estimation.

## Generative Adversarial Networks

## Autoregressive and Reversible Models

## Policy Gradient
   Q-Learning

A1. Loss Functions and Back-prop
   This assignment is meant to get your feet wet with computing the gradients for a model using backprop
   then translating your mathematical expressions into vectorized Python code
   Practice reasoning about the behavior of different loss functions.
   
   Binary classification task on MNIST dataset. 

