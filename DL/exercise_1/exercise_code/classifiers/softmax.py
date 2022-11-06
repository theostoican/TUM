"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    for i in range(0, X.shape[0]):
        gt_class = y[i]
        
        # Compute the scores for all classes
        linear_res = X[i].dot(W) #(1, C)
        class_scores = np.exp(linear_res) / np.sum(np.exp(linear_res))
        
        # Extract the probability of the correct class
        correct_class_score = class_scores[gt_class]
        
        # Compute the loss for the correct class
        loss -= np.log(correct_class_score) 
        
        # Compute the gradient
        
        # According to the gradient formulation, we must subtract one 
        # from the correct class
        class_scores[gt_class] -= 1
        for cls in range(dW.shape[1]):
            dW[:, cls] += X[i] * class_scores[cls]
    
    # Average on the batch size and add regularization
    loss = loss / X.shape[0] + reg / 2 * np.sum(W ** 2)
    dW = dW / X.shape[0] + reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # Compute softmax
    linear_res = X @ W #shape: N x C
    linear_res -= np.max(linear_res, axis = 1, keepdims=True)
    class_scores = np.exp(linear_res) / np.sum(np.exp(linear_res), axis = 1, keepdims=True)
    correct_class_scores = class_scores[range(X.shape[0]), y]

    # Compute loss
    loss = np.sum(-np.log(correct_class_scores)) / X.shape[0]
    loss += reg / 2 * np.sum(W ** 2)
    
    # Compute the gradient
    class_scores[range(X.shape[0]), y] -= 1
    # dW has shape D x C. Therefore for each column we should do:
        # dW[:, c] -= X[i] * class_scores[i][c] for each i
    # X has shape (N, D). class_scores has shape (N, C)
    dW = X.T @ class_scores / X.shape[0]
    dW += reg * W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [4.9e-7, 5e-7, 5.5e-7, 5.55e-7, 5.6e-7, 5.65e-7, 5.67e-7, 5.7e-7, 5.75e-7]
    #regularization_strengths = [2.5e4, 2.52e4, 2.55e4, 2.6e4, 2.65e4, 2.7e4]
    learning_rates = [10 ** i for i in np.linspace(-5.3, -5.2, num=5)]
    regularization_strengths = [10 ** i for i in np.linspace(4, 5, num=5)]
    learning_rates = np.linspace(5e-7, 5.25e-7, num=10)
    regularization_strengths = np.linspace(2.5e4, 2.55e4, num=10)
    learning_rates = [1e-7,1e-6,1e-5, 1e-4, 1e-3]
    regularization_strengths = [1e-1,1e1,5e1,1e2,1e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    num_iters = 2000
    for learn_rate in learning_rates:
        for reg_strength in regularization_strengths:
            # Create and train classifier
            classifier = SoftmaxClassifier()
            classifier.train(X_train, y_train, learning_rate = learn_rate, reg = reg_strength, num_iters=num_iters, verbose=False)
            
            # Evaluate classifier
            train_labels = classifier.predict(X_train)
            train_acc = np.mean(y_train == train_labels)
            
            val_labels = classifier.predict(X_val)
            val_acc = np.mean(y_val == val_labels)
            
            results[(learn_rate, reg_strength)] = (train_acc, val_acc)
            all_classifiers.append(classifier)
            
            if val_acc > best_val:
                best_val = val_acc
                best_softmax = classifier

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
