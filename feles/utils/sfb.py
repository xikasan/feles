# coding: utf-8

import numpy as np
import scipy.linalg as sl


def build_short_term_model(env):
    """
    Build short term mode of pitching motion of aircraft.
    |dT| = |  0   1||T| + | 0|de
    |dq|   |-Mw -Mq||q|   |de|
    :param env:
    :return:
    """
    model = env._model
    Ao = model._A.T
    Bo = model._B.T
    Mw = Ao[model.IX_q, model.IX_w]
    Mq = Ao[model.IX_q, model.IX_q]
    Mde = Bo[model.IX_q, model.IX_de]
    A = np.array([[0, 1], [Mw, Mq]], dtype=model.dtype)
    B = np.array([[0, Mde]], dtype=model.dtype).T
    return A, B


def compute_sfb_gain(env, Q, R):
    A, B = build_short_term_model(env)
    P = sl.solve_continuous_are(A, B, Q, R)
    K = sl.inv(R).dot(B.T).dot(P)
    return K
