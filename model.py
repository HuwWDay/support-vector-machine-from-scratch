"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # Calculate mean and std across columns (axis=0)
    means = np.mean(x, axis=0, keepdims=True)
    stds = np.std(x, axis=0, keepdims=True)
    
    # Where std is 0, replace it with 1 to avoid division by zero
    stds = np.where(stds == 0, 1, stds)
    
    return (x - means) / stds

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    return {"w": np.zeros(n_features), "b":0}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    return x @ params["w"] + params["b"]

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    return max(0, 1-y*score)

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    scores = compute_scores(x, params)
    loss = np.array([hinge_loss_example(scores[i], y[i]) for i in range(len(y))])
    return np.mean(loss)+reg_lambda* (params["w"] @ params["w"])

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    scores = compute_scores(x, params)
    m = 1-y*scores 
    mask = (m > 0).astype(float)
    dw = -x.T @ (mask*y)/x.shape[0]+2*reg_lambda*params["w"]
    db = -np.mean(mask*y)
    return {"dw":dw,"db":db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w, b = params["w"], params["b"]
    dw, db = grads["dw"], grads["db"]
    w_new = w- learning_rate*dw
    b_new = b-learning_rate*db
    return {"w":w_new, "b":b_new}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

