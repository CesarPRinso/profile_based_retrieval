import numpy as np
import itertools


def transform_pairwise(x, y):
    """Transforms data into pairs with balanced labels for ranking
       Transforms a n-class ranking problem into a two-class classification
       problem. Subclasses implementing particular strategies for choosing
       pairs should override this method.
       In this method, all pairs are choosen, except for those that have the
       same target value. The output is an array of balanced classes, i.e.
       there are the same number of -1 as +1
       Parameters
       ----------
       X : array, shape (n_samples, n_features)
           The data
       y : array, shape (n_samples,) or (n_samples, 2)
           Target labels. If it's a 2D array, the second column represents
           the grouping of samples, i.e., samples with different groups will
           not be considered.
       Returns
       -------
       X_trans : array, shape (k, n_feaures)
           Data as pairs
       y_trans : array, shape (k,)
           Output class labels, where classes have values {-1, +1}
       """
    x2 = []
    y2 = []

    for i in range(len(x)):
        for k in range(len(x)):
            if i == k or y[i, 0] == y[k, 0] or y[i, 1] != y[k, 1]:
                continue
            x2.append(x.iloc[i] - x.iloc[k])
            y2.append(np.sign(y[i, 0] - y[k, 0]))
            # output balanced classes
            if y2[-1] != (-1) ** k:
                y2[-1] = - y2[-1]
                x2[-1] = - x2[-1]
    return np.asarray(x2), np.asarray(y2)

