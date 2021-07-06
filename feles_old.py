# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller import compute_action as ca
from feles.controller.se import Log as sLog
from feles.controller.fel import Log as fLog
from feles.controller.fel import NLCFEL
from feles.utils.buffer import ReplayBuffer, FELESLog
from feles.utils.builder import *
from sye import retrieve as sretrieve
from fel import run_fel
from heritage.fel import run_fbc


def run(args):
    cf = xt.Config(args.config)
    if args.random:
        cf.env.reference.random = True

    # env
    env = build_env(cf.env)
    # LQR state feedback controller
    fbc = build_sfb(cf.controller.fbc, env)
    # fel controller
    fel = build_fel(cf.controller.fel, env)
    # system estimator
    sye = build_sye(cf.controller.sye, env)
    # reference queue
    ref = build_reference_queue(cf.env, at_random=cf.env.reference.random)

    ret_feles = run_feles(cf, env, ref, fbc, fel, sye)
    ret_fel = run_fel(cf, env, ref, fbc, fel)
    ret_fbc = run_fbc(cf, env, ref, fbc)
    plot(cf, env, ret_feles, ret_fel, ret_fbc)


def run_feles(cf, env, ref, fbc, fel, sye):
    env.reset()
    fel.reset()
    sye.reset()

    window_step = int(cf.controller.feles.window / cf.env.dt)
    bf_real = ReplayBuffer(window_step)
    bf_sim  = ReplayBuffer(window_step)
    log = xsim.Logger()

    past_x = None
    past_u = None
    past_r = None
    loss_sye = 0.
    loss_fel = 0.
    predict  = np.zeros_like(env.state)
    next_sample_time = cf.controller.feles.simulation.interval

    decay = build_decay_queue(window_step, cf.controller.sye.decay)
    cf.controller.feles.decay = decay

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()

        x = env.state
        u_fbc = ca.fbc(fbc, r, x)
        u_fel = ca.fel(fel, r)
        u = np.asanyarray(u_fbc + u_fel)
        u = np.squeeze(u)
        act = ca.build_action(env, u)

        if past_x is not None:
            bf_real.add(FELESLog(past_x, past_u, x, past_r))
            predict = sye.predict(past_x, past_u)

        if bf_real.is_full:
            data = bf_real.buffer
            loss_sye = sye.update(data.state, data.action, data.next_state, factor=decay)

        lr_factor_sim = np.exp(-cf.controller.feles.temp * loss_sye)
        do_extra_sampling = next_sample_time <= time and lr_factor_sim >= cf.controller.feles.threashold and False
        if do_extra_sampling:
            next_sample_time += cf.controller.feles.simulation.interval
            refs_sim = generate_ref_sim(cf, x[env.IX_T])
            x_sim = np.vstack([env.state,] * cf.controller.feles.simulation.num)
            for time_sim, r_sim in refs_sim.items():
                u_fbc_sim = sim_fbc(fbc, r_sim, x_sim)
                u_fel_sim = fel.get_actions(r_sim).reshape(-1, 1)
                u_sim = u_fbc_sim + u_fel_sim
                p_sim = sye.predicts(x_sim, u_sim)
                bf_sim.add(FELESLog(x_sim, u_sim, p_sim, r_sim))

                x_sim = p_sim

        if past_x is not None:
            data_real = bf_real.buffer
            r_train = data_real.reference
            u_train = data_real.action
            factor = np.ones_like(decay)
        else:
            r_train = r
            u_train = u
            factor = 1.

        if do_extra_sampling:
            data_sim = bf_sim.buffer
            r_train_sim = data_sim.reference
            u_train_sim = data_sim.action
            factor_sim = np.ones((r_train_sim.shape[0], 1)) * lr_factor_sim

            r_train = np.vstack([r_train, r_train_sim])
            u_train = np.vstack([u_train, u_train_sim])
            factor = np.vstack([factor, factor_sim])

        # loss_fel = np.mean([fel.update(r_train, u_train, factor) for k in range(5)], axis=0)
        # loss_fel = fel.update(r_train, u_train, factor)

        if isinstance(fel, NLCFEL):
            loss_fel, loss_lin, loss_nlc = loss_fel

        log.store(
            t=time,
            x=x,
            u=u,
            u_fbc=u_fbc,
            u_fel=u_fel,
            r=r,
            p=predict,
            l_sye=loss_sye,
            l_fel=loss_fel,
            f=lr_factor_sim,
        ).flush()

        env.step(act)
        past_x = x
        past_u = u
        past_r = r

    return xsim.Retriever(log)


def sim_fbc(fbc, r_sim, x_sim):
    e_sim = r_sim[..., :x_sim.shape[1]] - x_sim
    u_sim = fbc(e_sim.T).reshape((-1, 1))
    return u_sim


def generate_ref_sim(cf, x0):
    num_sim = cf.controller.feles.simulation.num
    r_range = np.array(cf.env.reference.range)
    r_range = xt.d2r(r_range)
    r_width = np.max(r_range) - np.min(r_range)
    dr = r_width / (num_sim - 1)
    rs = [dr * i for i in range(num_sim - 1)]
    rs.append(0.0)
    rs = np.array(rs) + np.min(r_range)
    r_filters = [xsim.Filter2nd(cf.env.dt, cf.env.reference.tau) for _ in rs]
    [r_filter.reset(x0) for r_filter in r_filters]

    def gen_single_ref(r, r_filter):
        r_filter(r)
        return r_filter.get_full_state()

    def generate_references():
        return np.array([
            gen_single_ref(r, r_filter)
            for r, r_filter in zip(rs, r_filters)
        ])

    refs = {
        time: generate_references()
        for time in xsim.generate_step_time(
            cf.controller.feles.simulation.due,
            cf.env.dt
        )
    }
    return refs


def retrieve(ret):
    ixT = 0
    r2d = xt.r2d
    res = pd.DataFrame(dict(
        time=ret.t(),
        reference=ret.r(ixT, fn=r2d),
        pitch=ret.x(ixT, fn=r2d),
        predict=ret.p(ixT, fn=r2d),
        loss_sye=ret.l_sye(),
        loss_fel=ret.l_fel(),
        u=ret.u(fn=r2d),
        u_fbc=ret.u_fbc(fb=r2d),
        u_fel=ret.u_fel(fb=r2d),
    ))
    return res


def plot(cf, env, res_feles, res_fel, res_fbc):
    xt.info("build result")
    # build results
    res_pitch = pd.DataFrame(dict(
        time=res_feles.t(),
        reference=res_feles.r(env.IX_T, fn=xt.r2d),
        feles=res_feles.x(env.IX_T, fn=xt.r2d),
        fel=res_fel.x(env.IX_T, fn=xt.r2d),
        fbc=res_fbc.x(env.IX_T, fn=xt.r2d)
    ))
    res_pred = pd.DataFrame(dict(
        time=res_feles.t(),
        pitch=res_feles.x(env.IX_T, fn=xt.r2d),
        speed=res_feles.x(env.IX_q, fn=xt.r2d),
        pitch_pred=res_feles.p(env.IX_T, fn=xt.r2d),
        speed_pred=res_feles.p(env.IX_q, fn=xt.r2d),
    ))

    res_error = pd.DataFrame(dict(
        time=res_feles.t(),
        feles=np.square(res_feles.r(env.IX_T, fn=xt.r2d) - res_feles.x(env.IX_T, fn=xt.r2d)),
        fel=np.square(res_fel.r(env.IX_T, fn=xt.r2d) - res_fel.x(env.IX_T, fn=xt.r2d)),
        fbc=np.square(res_fbc.r(env.IX_T, fn=xt.r2d) - res_fbc.x(env.IX_T, fn=xt.r2d))
    ))

    res_elevator = pd.DataFrame(dict(
        time=res_feles.t(),
        feles=res_feles.u(fn=xt.r2d),
        fel=res_fel.u(fn=xt.r2d),
        fbc=res_fbc.u(fn=xt.r2d),
    ))
    res_loss = pd.DataFrame(dict(
        time=res_feles.t(),
        loss_fel=res_feles.l_fel(),
        loss_sye=res_feles.l_sye(),
        factor=res_feles.f()
    ))


    xt.info("plot result")

    COLOR_ZERO = "black"
    WIDTH_ZERO = "1"

    C_REF = "gray"
    C_FBC = "dodgerblue"
    C_FEL = "forestgreen"
    C_FELES = "orangered"
    C_SET = [C_FBC, C_FEL, C_FELES]

    L_TIME = "time"
    L_SET = ["fbc", "fel", "feles"]

    fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(15, 12))

    ecf = cf.env
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res_pitch.plot(
        x=L_TIME, y="reference", ax=axes[0],
        style="--", color=C_REF,
        xlim=[0, ecf.due], grid=True,
        # ylim=ecf.reference.range,
        # yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1] + 1, 1)
    )
    res_pitch.plot(
        x=L_TIME, y=L_SET, ax=axes[0],
        style="-", color=C_SET
    )
    axes[0].set_ylabel("pitch [deg]")

    res_elevator.plot(
        x=L_TIME, y=L_SET, ax=axes[1],
        style="-", color=C_SET,
        # ylim=delim,
        grid=True
    )
    axes[1].set_ylabel("elevator [deg]")

    res_error.plot(
        x=L_TIME, y=L_SET, ax=axes[2], grid=True,
        color=C_SET
    )
    axes[2].set_ylabel("error [deg]")

    res_pred.plot(
        x="time", y=["pitch", "pitch_pred", "speed", "speed_pred"], ax=axes[3],
        style=["--", "-", "--", "-"], color=["forestgreen", "hotpink", "forestgreen", "hotpink"],
        grid=True
    )
    axes[3].set_ylabel("predict [deg]")

    res_loss.plot(
        x=L_TIME, y="loss_sye", ax=axes[4],
        grid=True
    )
    res_loss.plot(
        x=L_TIME, y="loss_fel", ax=axes[4],
        grid=True, secondary_y=True
    )
    axes[4].set_ylabel("loss")

    res_loss.plot(
        x=L_TIME, y="factor", ax=axes[5],
        grid=True
    )
    axes[5].set_ylabel("factor")

    if cf.result.fig.show:
        plt.show()
    if cf.result.fig.save:
        fig.savefig(xt.join(cf.result.path, "summary.png"))


def plot2(cf, res):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    L_TIME = "time"

    res.plot(
        x=L_TIME, y=["reference"], ax=axes[0],
        style="--", color="gray",
        xlim=[0, cf.env.due], ylim=cf.env.reference.range
    )
    res.plot(
        x=L_TIME, y=["pitch", "predict"], ax=axes[0]
    )

    res.plot(
        x=L_TIME, y="loss", ax=axes[1]
    )

    plt.show()



def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/feles.yaml")
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)

