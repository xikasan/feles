# coding: utf-8

import xair
import gym
import xai
import xsim
import numpy as np
import pandas as pd
import xtools as xt
import matplotlib.pyplot as plt
from feles.controller import se
from feles.controller import sfb
from feles.controller.fel import DeepFEL
from tqdm import tqdm


DTYPE = np.float32
DT = 0.01
DUE = 120

ENV_NAME = "LVAircraftPitch-v4"
ENV_TARGET_RANGE = xt.d2r([-1, 1])
ENV_TARGET_PERIOD = 10
ENV_FAIL_MODE = "GAIN_REDUCTION"
ENV_FAIL_RANGE = [0.5, 0.51]

# State Feedback Controller
Q = np.diag([100, 1]).astype(DTYPE)
R = np.diag([1]).astype(DTYPE)
# FEL Controller
# FEL_LR = 10
FEL_LR = 3e-4
FEL_UNITS = [8, 4]
# System Estimator
SYE_LR = 3e-4
SYE_UNITS = [8, 4]
L2_SCALE = 1e-2


def run():
    xt.info("run FELES control")
    xai.please()

    env = gym.make(
        ENV_NAME, dt=DT,
        target_range=ENV_TARGET_RANGE,
        target_period=ENV_TARGET_PERIOD,
        fail_mode=ENV_FAIL_MODE,
        fail_range=ENV_FAIL_RANGE
    )
    xt.info("env", env)
    env.reset()
    max_act = env.action_space.high[env.IX_de]
    dim_act = 1  # xai.get_size(env.action_space)
    dim_obs = xai.get_size(env.observation_space)
    dim_stt = dim_obs - 1
    dim_ref = env.get_target().shape[0]

    # controller
    # SFB
    K = sfb.compute_sfb_gain(env, Q, R).T
    K = np.hstack([np.zeros_like(K), K]).astype(DTYPE)
    # FEL
    fel = DeepFEL(FEL_UNITS, dim_act, max_act, lr=FEL_LR)
    fel.reset(dim_ref)
    xt.info("fel", fel)
    # SE
    sye = se.DeepSystemEstimator(SYE_UNITS, dim_stt, lr=SYE_LR, l2_scale=L2_SCALE)
    sye.reset(dim_stt, dim_act)
    xt.info("sye", sye)

    # logger
    log = xsim.Logger()

    # prepare one step before values
    past_stt = env.state
    past_act = np.zeros_like(env.state.dot(K))[[1]]

    for time in tqdm(xsim.generate_step_time(DUE, DT)):
        obs = env.observation
        stt = env.state

        act_sfb = compute_sfb_action(env, K)
        act_fel = compute_fel_action(env, fel)
        act = act_sfb + act_fel

        # for experiment record
        # there are not required to run FELES
        err = stt - env.target[:-1]
        pre = sye.predict(past_stt, past_act)

        # model updates
        if time == 60:
            online_simulation(env, sye)
        sye_loss = sye.update(past_stt, past_act, stt)
        fel_loss = fel.update(env.target, act[[env.IX_de]])
        log.store(
            time=time,
            obs=xt.r2d(obs),
            act=xt.r2d(act),
            err=xt.r2d(err),
            pre=xt.r2d(pre),
            sfb=xt.r2d(act_sfb),
            fel=xt.r2d(act_fel),
            sye_loss=sye_loss,
            fel_loss=fel_loss
        ).flush()

        # simulation update
        env.step(act)
        past_stt = stt
        past_act = act[[env.IX_de]]

    # result plot
    res = xsim.Retriever(log)
    res = pd.DataFrame({
        "time": res.time(),
        "reference": res.obs(idx=env.IX_C),
        "pitch": res.obs(idx=env.IX_T),
        "error": res.err(idx=env.IX_T),
        "predict": res.pre(idx=env.IX_T),
        "pitch_speed": res.obs(idx=env.IX_q),
        "error_speed": res.err(idx=env.IX_q),
        "predict_speed": res.pre(idx=env.IX_q),
        "elevator": res.act(idx=env.IX_de),
        "sfb": res.sfb(idx=env.IX_de),
        "fel": res.fel(idx=env.IX_de),
        "sye_loss": res.sye_loss(),
        "fel_loss": res.fel_loss(),
    })

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(DUE/5, 10))
    res.plot(x="time", y=["reference", "pitch", "predict"], ax=axes[0, 0], xlim=[0, DUE])
    res.plot(x="time", y=["fel_loss"], ax=axes[1, 0])
    res.plot(x="time", y=["elevator", "sfb", "fel"], ax=axes[2, 0])
    res.plot(x="time", y=["pitch_speed", "predict_speed"], ax=axes[0, 1])
    res.plot(x="time", y="sye_loss", ax=axes[1, 1])
    res.plot(x="time", y=["error"], ax=axes[2, 1])
    plt.show()


def compute_sfb_action(env, K):
    obs = env.observation
    x = obs[[env.IX_T, env.IX_q]]
    r = np.array([obs[env.IX_C], 0.], dtype=env.dtype)
    e = r - x
    u = e.dot(K)
    return u


def compute_fel_action(env, fel):
    ref = env.get_target()
    act = fel.get_action(ref)
    act = np.hstack([[0], act]).astype(DTYPE)
    return act


def predict_next_state(env, sye, stt, act):
    act = act[[env.IX_de]]
    pre = sye.predict(stt, act)
    return pre


def online_simulation(env, sye):
    # prepare states and reference
    stt = xt.bounded_normal(env.state.shape).astype(DTYPE)
    stt = xt.d2r(stt)
    act = np.zeros(1, dtype=DTYPE)
    tgt = np.random.choice(ENV_TARGET_RANGE)

    slog = xsim.Logger()

    for stime in xsim.generate_step_time(1, DT):
        print(stime, stt, tgt)
        stt = sye.predict(stt, act)
        slog.store(time=stime, state=stt).flush()

    sres = xsim.Retriever(slog)
    sres = pd.DataFrame({
        "time": sres.time(),
        "pitch": sres.state(idx=env.IX_T)
    })
    sres.plot(x="time", y="pitch", xlim=[0, 1])
    plt.show()
    exit()


if __name__ == '__main__':
    run()
