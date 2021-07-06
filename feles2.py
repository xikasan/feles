# coding: utf-8

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.utils.builder import *
from feles.utils.buffer import ReplayBuffer, FELESLog
from feles.controller import compute_action as ca
from heritage.fel import run_fel, run_fbc


COLOR_ZERO = "black"
COLOR_DATA = ["gray", "forestgreen", "hotpink"]
WIDTH_ZERO = "1"


def run(args):
    xt.info("run feles")
    cf = xt.Config(args.config)

    env = build_env(cf.env)
    fbc = build_sfb(cf.controller.sfb, env)
    # fel = build_fel(cf.controller.fel, env) if not args.deep else build_dfel(cf.controller.dfel, env)
    fel = build_dfel(cf.controller.dfel, env)
    sye = build_sye(cf.controller.sye, env)
    ref = build_reference_queue(cf.env, at_random=False)

    save(cf.result, None, None, None)

    res_feles = run_feles(cf, env, ref, fbc, fel, sye)
    fel = build_fel(cf.controller.fel, env)
    res_fel = run_fel(cf, env, ref, fbc, fel)
    res_fbc = run_fbc(cf, env, ref, fbc)

    # make directory
    if cf.result.fig.save or cf.result.data:
        cf.result.path = xt.make_dirs_current_time(cf.result.path)
        cf.dump(xt.join(cf.result.path, "config.yaml"))

    # make figure
    if cf.result.fig.save or cf.result.fig.show:
        plot(cf, env, res_feles, res_fel, res_fbc)
    # save data sa csv file
    if cf.result.data:
        save(cf.result.path, res_feles, res_fel, res_fbc)


def run_feles(cf, env, ref, fbc, fel, sye):
    env.reset()
    fel.reset()
    log = xsim.Logger()
    bf_real = ReplayBuffer(int(cf.controller.sye.window / cf.env.dt))
    bf_sim = ReplayBuffer(int(cf.sim.due / cf.sim.dt))

    past_x = None
    past_u = None
    loss_sye = 0.0
    loss_fel = 0.0
    lr_factor_sim = 0.
    predict = np.zeros_like(env.state)
    ref_queue = ref

    do_extra_sampling = False

    cf.controller.feles.decay = np.array([cf.controller.feles.decay]) ** np.arange(bf_sim.max_size).reshape((bf_sim.max_size, 1))

    for time, r in tqdm(ref_queue.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u_fbc = ca.fbc(fbc, r, x)
        u_fel = ca.fel(fel, r)
        u = np.asanyarray(u_fbc + u_fel)
        u = np.squeeze(u)
        act = ca.build_action(env, u)

        # L1) Control loop sampling
        if past_x is not None:
            bf_real.add(FELESLog(past_x, past_u, x, r))
            predict = sye.predict(past_x, past_u)

        # A1) System Estimation
        if len(bf_real) >= bf_real.max_size:
            data = bf_real.buffer
            loss_sye = np.mean([
                sye.update(data.state, data.action, data.next_state)
                for i in range(cf.controller.sfb.repeat)
            ])

            # L2) Simulation loop sampling
            lr_factor_sim = np.exp(-cf.controller.feles.temp * loss_sye)
            do_extra_sampling = lr_factor_sim >= cf.controller.feles.threashold

        if do_extra_sampling:
            ref_sim = build_reference_queue(cf.sim)
            x_sim = ref_sim[0.][[env.IX_T, env.IX_q]]

            for time_sim, r_sim in ref_sim.items():
                u_fbc_sim = ca.fbc(fbc, r_sim, x_sim)
                u_fel_sim = ca.fel(fel, r_sim)
                u_sim = np.asanyarray(u_fbc_sim + u_fel_sim)

                p_sim = sye.predict(x_sim, u_sim)
                bf_sim.add(FELESLog(x_sim, u_sim, p_sim, r_sim))

                x_sim = p_sim

        # A2) FEL adaptation
        if past_x is not None:
            data_real = bf_real.buffer
            r_train = data_real.reference
            u_train = data_real.action
            factor = cf.controller.feles.decay[:r_train.shape[0], ...]
        else:
            r_train = r
            u_train = u
            factor = 1.

        if do_extra_sampling:
            data_sim = bf_sim.buffer
            r_train_sim = data_sim.reference
            u_train_sim = data_sim.action
            factor_sim = np.ones((r_train_sim.shape[0], 1)) * lr_factor_sim * 0.2

            r_train = np.vstack([r_train, r_train_sim])
            u_train = np.vstack([u_train, u_train_sim])
            factor = np.vstack([factor, factor_sim])

        loss_fel = np.mean([fel.update(r_train, u_train, factor) for i in range(cf.controller.fel.repeat)])

        log.store(
            t=time,
            r=r,
            x=x,
            u=u,
            u_fbc=u_fbc,
            u_fel=u_fel,
            p=predict,
            l_fel=loss_fel,
            l_sye=loss_sye,
            f=lr_factor_sim
        ).flush()

        past_x = x
        past_u = u
        env.step(act)

    return xsim.Retriever(log)


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
        xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
        yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1] + 1, 1)
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


def save(path, res_feles=None, res_fel=None, res_fbc=None):
    if res_feles is not None:
        res_feles = build_feles_result(res_feles)
        res_feles.to_csv(xt.join(path, "feles.csv"))
    if res_feles is not None:
        res_fel = build_fel_result(res_fel)
        res_fel.to_csv(xt.join(path, "fel.csv"))
    if res_feles is not None:
        res_fbc = build_fbc_result(res_fbc)
        res_fbc.to_csv(xt.join(path, "fbc.csv"))


def build_fbc_result(ret):
    r2d = xt.r2d
    ixT = 0
    ixq = 1
    return pd.DataFrame(dict(
        time=ret.t(),
        reference=ret.r(ixT, fn=r2d),
        reference_speed=ret.r(ixq, fn=r2d),
        pitch=ret.x(ixT, fn=r2d),
        pitch_speed=ret.x(ixq, fn=r2d),
        error=np.abs(ret.r(ixT, fn=r2d) - ret.x(ixT, fn=r2d)),
        elevator=ret.u(fn=r2d)
    ))


def build_fel_result(ret):
    res = build_fbc_result(ret)
    res["loss_fel"] = ret.l_fel()
    res["u_fel"] = ret.u_fel(fn=xt.r2d)
    res["u_fbc"] = ret.u_fbc(fn=xt.r2d)
    return res


def build_feles_result(ret):
    res = build_fel_result(ret)
    res["predict_pitch"] = ret.p(0, fn=xt.r2d)
    res["predict_speed"] = ret.p(1, fn=xt.r2d)
    res["factor"] = ret.f()
    return res


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feles_normal.yaml")
    parser.add_argument("--deep", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arguments()
    run(args)
