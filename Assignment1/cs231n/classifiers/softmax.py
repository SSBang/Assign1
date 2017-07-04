import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    num_train = X.shape[0]
    num_labels = W.shape[1]

    S = np.dot(X, W)

    for i in range(num_train):
        sum_ith_score = np.sum(np.exp(S[i]))
        true_score = np.exp(S[i, y[i]])
        loss += -np.log(true_score / sum_ith_score)

    loss /= num_train
    loss += reg * np.sum(W * W)

    exp_S = np.exp(S)

    rowsum_exp_S = np.sum(exp_S, axis=1)

    rowsum_exp_S_reshape = rowsum_exp_S.reshape((X.shape[0], 1))
    dS = exp_S / rowsum_exp_S_reshape

    labeled_S = np.zeros_like(S)
    labeled_S[np.arange(X.shape[0]), y] = -1

    dS += labeled_S
    dS /= num_train

    dW = np.dot(X.T, dS) + 2 * reg * W
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    num_train = X.shape[0]
    num_labels = W.shape[1]

    S = np.dot(X, W)
    max_S = np.amax(S)
    exp_S = np.exp(S - max_S)  # 책에서 말한 inf 제거 방법
    rowsum_exp_S = np.sum(exp_S, axis=1)
    true_labed_exp_S = exp_S[np.arange(num_train), y]
    margin = -np.log(true_labed_exp_S / rowsum_exp_S)
    loss = np.sum(margin)

    loss /= num_train
    loss += reg * np.sum(W * W)

    # dW

    rowsum_exp_S_reshape = rowsum_exp_S.reshape((X.shape[0], 1))
    dS = exp_S / rowsum_exp_S_reshape

    labeled_S = np.zeros_like(S)
    labeled_S[np.arange(X.shape[0]), y] = -1

    dS += labeled_S
    dS /= num_train

    dW = np.dot(X.T, dS) + 2 * reg * W
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW
