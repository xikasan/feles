# coding: utf-8

import xsim
import numpy as np
import xtools as xt


def build_reference_queue(dt, due, ref_cf):
    rcf = ref_cf
    reference_width = (np.max(rcf.range) - np.min(rcf.range)) / 2
    reference_width = xt.d2r(reference_width) * 2
    ref = xsim.PoissonRectangularCommand(
        max_amplitude=reference_width,
        interval=rcf.interval
    )
    ref.reset()
    ref_filter = xsim.Filter2nd(dt, rcf.tau)
    ref_filter.reset()
    xt.info("reference generator", ref)

    def generate_full_state_reference(t):
        ref_filter(ref(t))
        return ref_filter.get_full_state()

    ref = {
        time: generate_full_state_reference(time)
        for time in xsim.generate_step_time(due, dt)
    }
    return ref
