# coding: utf-8


import numpy as np
from feles.controller.fel import FEL


def run():
    dim_ref = 3
    dim_act = 2
    max_act = 1
    fel = FEL(dim_ref, dim_act, max_act)
    print(fel.lout.weight)

    dummy_ref = np.random.rand(3)
    act = fel.get_action(dummy_ref)
    print(act)

    print("+"*60)
    dummy_act = np.random.rand(2)
    for _ in range(1000):
        loss = fel.update(dummy_ref, dummy_act)
        print("[loss]", loss)
    print(fel.lout.weight)
    fel.reset()
    print(fel.lout.weight)


if __name__ == '__main__':
    run()
