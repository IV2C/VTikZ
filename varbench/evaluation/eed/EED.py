# -*- coding:utf-8 -*-
import ctypes
import os
import sys

print(sys.path)
from varbench.evaluation.eed import util


# Python wrpaper for the C++ EED implementation
def eed(hyp_str: str, ref_str: str):

    hyp = list(hyp_str)
    ref = list(ref_str)

    _eed = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__)) + "/libEED.so")
    _eed.wrapper.restype = ctypes.c_float
    hyp.insert(0, " ")
    hyp.append(" ")
    ref.insert(0, " ")
    ref.append(" ")
    hyp_c = (ctypes.c_ulonglong * len(hyp))()
    hyp_c[:] = [bytes_to_int(x.encode("utf-8")) for x in hyp]
    ref_c = (ctypes.c_ulonglong * len(ref))()
    ref_c[:] = [bytes_to_int(x.encode("utf-8")) for x in ref]
    alpha = 2.0
    deletion = 0.2
    insertion = 1.0
    substitution = 1.0
    rho = 0.3
    norm = len(ref_c)
    result = _eed.wrapper(
        hyp_c,
        ref_c,
        len(hyp_c),
        len(ref_c),
        ctypes.c_float(alpha),
        ctypes.c_float(deletion),
        ctypes.c_float(insertion),
        ctypes.c_float(substitution),
        ctypes.c_float(rho),
        norm,
    )
    return min(1.0, 1.0 * result)


def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result
