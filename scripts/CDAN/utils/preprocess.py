import numpy as np

def normalize_source_target(Xs, Xt):
    mean = np.mean(Xs)
    std = np.std(Xs)

    Xs = (Xs - mean) / std
    Xt = (Xt - mean) / std

    return Xs, Xt, mean, std

def normalize_test(X, mean, std):
    return (X - mean) / std