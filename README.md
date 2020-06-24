# Intro-to-Neural-Network
A review of materials for the senior university level machine learning course. 

Prerequisites: Advanced calculus | Object-Oriented-Programming | Python3.6

Computing Technique: Backpropagation Algorithm (For gradient) | Automatic Differentiation

Reference Page: http://www.cs.toronto.edu/~rgrosse/

# Selected Fundamental Topics
## 1. Supervised Learning vs. Unsupervised Learning
  - The difference, and typical tasks
  - supervised learning:  predict with a given target
      - linear regression: predict a scalar valued target
      - linear classification: classify the result
  - unsupervised learning: predict with unspecified target

  - core idea behind Supervised Learning Program:
    - training set (contain part of both X and Y, need to concatenate explainatory dataset and target dataset)
    - testing set / validation set
    * consideration in the split of dataset: generalization problem
    
2. Neural Network: Multilayer perceptrons
  - perceptron: 
    Comparison of activation functions. 
    Viewing deep neural nets as function composition and as feature learning. 
    Limitations of linear networks and universality of nonlinear networks


3. ML models[Problems related to Supervised Learning]
Models
  - Linear Regression: 
      - Can be formulated as an optimization problem, 
      - can be solved
                - directly
                - gradient descent [algorithm]
      - 

* Speed Booster Techniques / Thought Process:
  - Vectorization
  - Feature maps and polynomial regression
  - Generalization: overfitting, underfitting, and validation.

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
  - How to use the gradients computed by backprop. 
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
    Equivariance and invariance. 
    Backprop rules for conv nets.

Image Classification
    Conv net architectures applied to handwritten digit and object classification. 
    Measuring the size of a conv net.

Optimizing the Input
    Interesting things you can do with gradient descent on the inputs: 
     - conv net visualizations, adversarial inputs, Deep Dream.

Recurrent Neural Nets
  - Backprop through time. Applying RNNs to language modeling and machine translation.
Learning Long-Term Dependencies
  - Why RNN gradients explode and vanish
    - the mechanics of backprop
    - conceptually in terms of the function the RNN computes 
  - Ways to deal with it: gradient clipping, input reversal, LSTM

ResNets and Attention
    Deep Residual Networks. 
    Attention-based models for machine translation and caption generation.

Learning Probabilistic Models
    Maximum likelihood estimation. 
    basics of Bayesian parameter estimation and maximum a-posteriori estimation.

Generative Adversarial Networks


Autoregressive and Reversible Models
Policy Gradient
Q-Learning
Go

