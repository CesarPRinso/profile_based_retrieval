from sklearn import svm
from pair import transform_pairwise

def r_train(x, y):
    """
    Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    x2, y2 = transform_pairwise(x, y)
    svc = svm.SVC(kernel='linear',
                  probability=True,
                  max_iter=10000000).fit(x2, y2)
    return svc
