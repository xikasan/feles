# coding: utf-8

import xsim
import xtools as xt
import pandas as pd


def def_dataframe(res):
    return res if res is not None else pd.DataFrame()


def base(env, log, res=None):
    ret = xsim.Retriever(log)
    res = def_dataframe(res)
    res["time"] = res.t()
    res["reference"] = ret.r(env.IX_T, fn=xt.r2d)
    return res


def pitch(env, log, res=None):
    ret = xsim.Retriever(log)
    res = def_dataframe(res)
    res["pitch"] = ret.x(env.IX_T)
    return res


def fel(env, log, res=None):
    ret = xsim.Retriever(log)
    res = def_dataframe(res)
