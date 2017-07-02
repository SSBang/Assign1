import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    ds = np.zeros((num_train,num_classes))
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                ds[i,j] = 1  # 정답 label이 아니고 margin이 양수이면 1 
                ds[i,y[i]] -= 1  # 정답 label은 margin이 양수일 때만 -1 , 음수일 때는 0이므로

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    ds /= num_train



    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW = np.dot(X.T,ds) + 2 * reg * W 
    

    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    ##########################################################################
    number_classes = W.shape[1]
    number_train = X.shape[0]
    number_dim = X.shape[1]
    
    # Loss 
    S = np.dot(X, W)
    select_label = S[np.arange(number_train), y]
    select_label = select_label.reshape((number_train, 1))

    margins = S - select_label + 1
    margins_max = np.maximum(margins, 0)
    margins_max[np.arange(number_train), y] = 0
    
    L = np.sum(margins_max)/number_train
    loss = L + reg * np.sum(W * W)

    # gradient
    label_positive = np.zeros_like(margins)
    label_positive[margins_max > 0] = 1
    ds_true = np.sum(label_positive, axis=1)
    ds_true *= -1
    
    ds = np.ones_like(S)
    ds[margins < 0] = 0
    ds[np.arange(number_train), y] = ds_true 
    ds /= number_train
    
    dW = np.dot(X.T, ds) + 2 * reg * W
    

 
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW
