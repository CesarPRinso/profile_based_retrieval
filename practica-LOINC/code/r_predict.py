import numpy as np
from tool import sort


def r_predict(svc, x):
    """
            Predict an ordering on X. For a list of n samples, this method
            returns a list from 0 to n-1 with the relative order of the rows of X.
            The item is given such that items ranked on top have are
            predicted a higher ordering (i.e. 0 means is the last item
            and n_samples would be the item ranked on top).
            Parameters
            ----------
            X : array, shape (n_samples, n_features)
            Returns
            -------
            ord : array, shape (n_samples,)
                Returns a list of integers representing the relative order of
                the rows in X.
            """
    s = []
    for i in range(len(x)):
        t = []
        for k in range(len(x)):
            if i != k:
                t.append(x.iloc[i] - x.iloc[k])
        s.append(sum(svc.predict(np.asarray(t))))

    return sort(np.asarray(s))
