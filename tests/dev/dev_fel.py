# coding: utf-8

import xair
import gym
import xsim
import numpy as np
import xtools as xt
import pandas as pd
import matplotlib.pyplot as P
from tqdm import tqdm

from feles.controller.fel import FEL


cf = xt.Config(dict(
    dt=0.01,
    due=10,
    env="LVAircraftPitch-v4",
))

def run():
    env = gym.make("LVAircraftPitch-v4", dt=0.01)
