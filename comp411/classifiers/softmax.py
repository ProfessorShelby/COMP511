import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    bs = X.shape[0]

    for i in range(bs):
        logits = X[i] @ W
        ll = np.exp(logits - logits.max())
        softmax = ll / ll.sum()
        loss -= np.log(softmax[y[i]])  # nll
        softmax[y[i]] -= 1  # update for gradient
        dW += X[i,:].reshape(-1,1) @ softmax.reshape(1,-1)

    loss /= bs
    dW /= bs

    if regtype == 'L2':
        loss = loss + reg_l2 * (W ** 2).sum()
        dW = dW + 2 * reg_l2 * W
    else:
        loss = loss + reg_l2 * (W ** 2).sum() + reg_l1 * np.abs(W).sum()
        dW = dW + 2 * reg_l2 * W + reg_l1 * np.sign(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    bs = X.shape[0]

    logit = X @ W
    probs = np.exp(logit) / np.sum(np.exp(logit), axis=1).reshape(-1, 1)

    loss = np.sum(-np.log(probs[range(bs), y]))
    loss /= bs

    dprobs = probs
    dprobs[range(bs), y] -= 1
    dW = X.T @ dprobs
    dW /= bs

    if regtype == 'L2':
        loss += reg_l2 * np.sum(W**2)
        dW += 2 * reg_l2 * W
    else:
        loss += reg_l2 * np.sum(W**2) + reg_l1 * np.abs(W).sum()
        dW += 2 * reg_l2 * W + reg_l1 * np.sign(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
