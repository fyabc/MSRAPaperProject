# Reinforcement Learning for Machine Learning

by Fei Tian


## Abstract

Using policy gradient algorithm to help optimizing the training of NN.

-------------

For each mini-batch of input of NN, we want to learn a mask (an array of 0/1).
If `mask[i] == 0`, it means we should discard `input[i]`, else we should use this input.


## Elements of Reinforcement Learning

One minibatch is a sequence of input in RL.

- State
    mask array + the softmax probability of the output layer of NN.
    
    the output of NN may describe the state of it:
    >   if the cross entropy of output is very high (almost equal probabilities),
        this input may be useless, so we want to set the mask of it to 0.
        
- Action
    the mask array.
    
- Reward
    Immediate reward or final reward.
    Using the loss value or others.
    
    
## Policy Network

It is just a 2-classify problem, using logistic regression.

>   `action = sigmoid(w * state + b)`
    we want to learn `w` and `b`.
