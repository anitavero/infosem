import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/."))
from embedding import *


def test_order_local():
    V = np.array([[1, 2, 3], [4, 5, 6]])
    Vt = np.empty((2, 3, 5))
    Vt[:, :, 4] = V * 3
    Vt[:, :, 3] = V * 2
    Vt[:, :, 2] = V + 5
    Vt[:, :, 1] = V + 0.5
    Vt[:, :, 0] = V

    lorder = order_local(Vt, 1)

    assert len(lorder) == 4
    assert lorder[0] == cosine_similarity(np.array([0.5, 0.5, 0.5]).reshape(1, -1),
                                          np.array([0.5, 0.5, 0.5]).reshape(1, -1))
    assert lorder[1] == cosine_similarity(np.array([4.5, 4.5, 4.5]).reshape(1, -1),
                                          np.array([4.5, 4.5, 4.5]).reshape(1, -1))
    assert lorder[2] == cosine_similarity(np.array([2-6, 4-7, 6-8]).reshape(1, -1),
                                          np.array([8-9, 10-10, 12-11]).reshape(1, -1))


    V = np.array([[1, 2, 3], [40, 50, 60], [100, 200, 300]])
    Vt = np.empty((3, 3, 2))
    Vt[:, :, 1] = V + 0.5
    Vt[0, :, 1] = [11, 12, 13]
    Vt[:, :, 0] = V

    lorder = order_local(Vt, 1)

    assert len(lorder) == 1
    # Velocities of
    # Vt[0] . Vt[1]
    # Vt[1] . Vt[0]
    # Vt[2] . Vt[1]
    assert lorder[0] == (cosine_similarity(np.array([10, 10, 10]).reshape(1, -1),
                                           np.array([0.5, 0.5, 0.5]).reshape(1, -1))   + \
                         cosine_similarity(np.array([0.5, 0.5, 0.5]).reshape(1, -1),
                                           np.array([10, 10, 10]).reshape(1, -1))      + \
                         cosine_similarity(np.array([0.5, 0.5, 0.5]).reshape(1, -1),
                                           np.array([0.5, 0.5, 0.5]).reshape(1, -1)))    \
                        / 3