# coding: utf-8

import xsim
import xtools as xt
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from feles.controller import se
from feles.controller import compute_action as ca
from feles.utils.builder import *
from feles.utils.buffer import ReplayBuffer
from feles.controller.se import NLCSystemEstimator


def run(args):
    xt.info("dev of NLCSystemEstimator")
    cf = xt.Config(args.config)

    # prepare env
    env = build_env(cf.env)
    # state feedback controller
    fbc = build_sfb(cf.controller.sfb, env)
    # prepare reference
    ref = build_reference_queue(cf.env, at_random=True)
    # system estimator with non-linear complement
    scf = cf.controller.nse
    dim_state = env.state.shape[0]
    dim_action = 1
    nse = NLCSystemEstimator(
        dim_state, dim_action, scf.units,
        lr_ssl=scf.lr.ssl, lr_nlc=scf.lr.nlc
    )

    sye = build_sye(cf.controller.sye, env, dim_action)

    ret_sye = run_nse(cf, env, ref, fbc, sye)
    ret_nse = run_nse(cf, env, ref, fbc, nse)

    plot(cf, env, ret_sye, ret_nse)


def run_nse(cf, env, ref, fbc, nse):
    env.reset()
    nse.reset()
    log = xsim.Logger()
    bfr = ReplayBuffer(int(cf.controller.sye.window / cf.env.dt))

    past_x = None
    past_u = None
    loss = 0.0
    predict = np.zeros_like(env.state)

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u = ca.fbc(fbc, r, x)
        act = ca.build_action(env, u)

        if past_x is not None:
            bfr.add(se.Log(past_x, past_u, x))
            predict = nse.predict(past_x, past_u)

        if bfr.is_max:
            data = bfr.buffer
            loss = nse.update(data.state, data.action, data.next_state)

        log.store(
            t=time,
            r=r,
            x=x,
            u=u,
            p=predict,
            l=loss,
            ws=nse.W.flatten()
        ).flush()

        past_x = x
        past_u = u
        env.step(act)

    return xsim.Retriever(log)


def plot(cf, env, ret_sye, ret_nse):
    res = pd.DataFrame(dict(
        time=ret_nse.t(),
        reference=ret_nse.r(env.IX_T, fn=xt.r2d),
        pitch=ret_nse.x(env.IX_T, fn=xt.r2d),
        predict=ret_nse.p(env.IX_T, fn=xt.r2d),
        speed=ret_nse.x(env.IX_q, fn=xt.r2d),
        pred_speed=ret_nse.p(env.IX_q, fn=xt.r2d),
        loss=ret_nse.l(),
        w1=ret_nse.ws(0),
        w2=ret_nse.ws(1),
        w3=ret_nse.ws(2),
        w4=ret_nse.ws(3),
        w5=ret_nse.ws(4),
        w6=ret_nse.ws(5),
    ))

    res_pitch = pd.DataFrame(dict(
        time=ret_nse.t(),
        reference=ret_nse.r(env.IX_T, fn=xt.r2d),
        pitch=ret_nse.x(env.IX_T, fn=xt.r2d),
        nse=ret_nse.p(env.IX_T, fn=xt.r2d),
        sye=ret_sye.p(env.IX_T, fn=xt.r2d),
    ))

    res_speed = pd.DataFrame(dict(
        time=ret_nse.t(),
        speed=ret_nse.x(env.IX_q, fn=xt.r2d),
        nse=ret_nse.p(env.IX_q, fn=xt.r2d),
        sye=ret_sye.p(env.IX_q, fn=xt.r2d),
    ))

    res_error = pd.DataFrame(dict(
        time=ret_nse.t(),
        nse_T=np.abs(ret_nse.x(env.IX_T, fn=xt.r2d) - ret_nse.p(env.IX_T, fn=xt.r2d)),
        nse_q=np.abs(ret_nse.x(env.IX_q, fn=xt.r2d) - ret_nse.p(env.IX_q, fn=xt.r2d)),
        sye_T=np.abs(ret_sye.x(env.IX_T, fn=xt.r2d) - ret_sye.p(env.IX_T, fn=xt.r2d)),
        sye_q=np.abs(ret_sye.x(env.IX_q, fn=xt.r2d) - ret_sye.p(env.IX_q, fn=xt.r2d))
    ))
    res_loss = pd.DataFrame(dict(
        time=ret_nse.t(),
        nse=ret_nse.l(),
        sye=ret_sye.l(),
    ))

    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(10, 8))
    res_pitch.plot(
        x="time", y=["reference", "pitch", "sye", "nse"], ax=axes[0],
        xlim=[0, cf.env.due], ylim=cf.env.reference.range, grid=True
    )
    res_speed.plot(
        x="time", y=["speed", "sye", "nse"], ax=axes[1],
        xlim=[0, cf.env.due], ylim=cf.env.reference.range, grid=True
    )
    res_error.plot(
        x="time", y=["nse_T", "sye_T", "nse_q", "sye_q"], ax=axes[2], grid=True,
        # xlim=[0, cf.env.due], ylim=cf.env.reference.range, grid=True
        style=["--", "--", "-", "-"], color=["forestgreen", "orangered", "forestgreen", "orangered"]
    )
    res_loss.plot(
        x="time", y=["sye", "nse"], ax=axes[3], grid=True,
        # xlim=[0, cf.env.due], ylim=cf.env.reference.range, grid=True
        color=["forestgreen", "orangered"]
    )


    # res.plot(
    #     x="time", y=["reference", "pitch", "predict"], ax=axes[0],
    #     style=["--", "-", "-"], color=["gray", "forestgreen", "hotpink"],
    #     xlim=[0, cf.env.due], ylim=cf.env.reference.range, grid=True
    # )
    # res.plot(
    #     x="time", y=["speed", "pred_speed"], ax=axes[1],
    #     style=["-", "-"], color=["forestgreen", "hotpink"],
    #     grid=True
    # )
    #
    # res.plot(
    #     x="time", y="loss", ax=axes[2],
    #     grid=True
    # )
    #
    # res.plot(
    #     x="time", y=["w1", "w2", "w3", "w4", "w5", "w6"],
    #     ax=axes[3], grid=True
    # )

    plt.show()


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nse_normal.yaml")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)
