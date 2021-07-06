# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller import compute_action as ca
from feles.controller.fel import Log, NLCFEL
from feles.utils.buffer import ReplayBuffer
from feles.utils.builder import *
from fel import run_fel, retrieve


def run(args):
    cf = xt.Config(args.config)
    if args.random:
        cf.env.reference.random = True

    # env
    env = build_env(cf.env)
    # LQR state feedback controller
    fbc = build_sfb(cf.controller.fbc, env)
    # reference queue
    ref = build_reference_queue(cf.env, at_random=cf.env.reference.random)

    fcf = cf.controller.fel

    lrs = [
        xt.round([(i_lin + 1) * 0.05, (i_nlc + 1) * 0.05], 6)
        for i_lin in range(20) for i_nlc in range(20)
    ]

    reses = []
    for lr in lrs:
        fcf.lr_lin = lr[0]
        fcf.lr_nlc = lr[1]
        fel = build_fel(cf.controller.fel, env)
        ret = run_fel(cf, env, ref, fbc, fel)
        res = retrieve(ret)
        reses.append(res)

    losses = [
        np.mean(res.loss)
        for res in reses
    ]

    for lr, loss in zip(lrs, losses):
        print("- "*60)
        print("lr:", lr)
        print("loss:", loss)
    print("="*120)
    min = np.argmin(losses)
    print("lr:", lrs[min])
    print("loss:", losses[min])
    # 0.45-0.05


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fel.yaml")
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)
