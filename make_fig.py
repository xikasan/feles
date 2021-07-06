# coding: utf-8

import xtools as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
COLOR_ZERO = "black"
WIDTH_ZERO = "1"

C_REF = "gray"
C_FBC = "dodgerblue"
C_FEL = "forestgreen"
C_FELES = "orangered"
C_SET = [C_FBC, C_FEL, C_FELES]

L_TIME = "time"
L_SET = ["fbc", "fel", "feles"]


def rect():
    # source = "result/champion/random"
    # source = "result/champion/gain3.r3"
    source = "result/champion/champion"

    feles = xt.join(source, "feles.csv")
    fel = xt.join(source, "fel.csv")
    fbc = xt.join(source, "fbc.csv")

    feles = pd.read_csv(feles)
    fel = pd.read_csv(fel)
    fbc = pd.read_csv(fbc)

    res_pitch = pd.DataFrame()
    res_pitch["time"] = fel.time
    res_pitch["reference"] = fel.reference
    res_pitch["feles"] = feles.pitch
    res_pitch["fel"] = fel.pitch
    res_pitch["fbc"] = fbc.pitch

    res_elev = pd.DataFrame()
    res_elev["time"] = fel.time
    res_elev["feles"] = feles.elevator
    res_elev["fel"] = fel.elevator
    res_elev["fbc"] = fbc.elevator

    res_error = pd.DataFrame()
    res_error["time"] = fel.time
    res_error["feles"] = feles.error ** 2
    res_error["fel"] = fel.error ** 2
    res_error["fbc"] = fbc.error ** 2

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))

    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res_pitch.plot(
        x=L_TIME, y="reference", ax=axes[0],
        color=C_REF, style="--",
        xlim=[0., 240.]
    )
    res_pitch.plot(
        x=L_TIME, y=L_SET, ax=axes[0],
        color=C_SET,
        # ylim=[-2, 2],
        grid=True
    )
    axes[0].set_ylabel("Pitch [deg]")
    axes[0].axvline(120, color="red", linewidth=WIDTH_ZERO)

    res_elev.plot(
        x=L_TIME, y=L_SET, ax=axes[1],
        color=C_SET,
        # ylim=[-5, 5],
        grid=True, legend=False
    )
    axes[1].set_ylabel("Elevator deflection [deg]")
    axes[1].axvline(120, color="red", linewidth=WIDTH_ZERO)

    res_error.plot(
        x=L_TIME, y=L_SET, ax=axes[2],
        color=C_SET, grid=True, legend=False
    )
    axes[2].set_ylabel("Control Error [deg]")
    axes[2].axvline(120, color="red", linewidth=WIDTH_ZERO)
    axes[2].set_xlabel("time [sec]")

    plt.show()


def estimate():
    # source = "result/champion/random"
    # source = "result/champion/gain3.r3"
    source = "result/champion/champion"

    feles = xt.join(source, "feles.csv")
    feles = pd.read_csv(feles)

    res_pitch = pd.DataFrame()
    res_pitch["time"] = feles.time
    res_pitch["observation"] = feles.pitch
    res_pitch["predict"] = feles.predict_pitch

    res_speed = pd.DataFrame()
    res_speed["time"] = feles.time
    res_speed["observation"] = feles.pitch_speed
    res_speed["predict"] = feles.predict_speed


    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 5))

    L_SET = ["observation", "predict"]
    C_SET = ["black", "orangered"]
    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    res_pitch.plot(
        x=L_TIME, y=L_SET, ax=axes[0],
        color=C_SET, xlim=[0, 240], grid=True
    )
    axes[0].set_ylabel("Pitch [deg]")
    axes[0].axvline(120, color="red", linewidth=WIDTH_ZERO)

    res_speed.plot(
        x=L_TIME, y=L_SET, ax=axes[1],
        color=C_SET, xlim=[0, 240], legend=False, grid=True
    )
    axes[1].set_ylabel("Pitch speed [deg/s]")
    axes[1].axvline(120, color="red", linewidth=WIDTH_ZERO)

    feles.plot(x=L_TIME, y="factor", ax=axes[2], grid=True, color="orangered")
    axes[2].set_ylabel("Training factor")

    axes[-1].set_xlabel("time [sec]")

    plt.show()




if __name__ == '__main__':
    rect()
    estimate()
