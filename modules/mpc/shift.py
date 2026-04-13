import numpy as np

def shift(T, t0, x0, u0, f):
    # 현재 상태와 첫 번째 입력으로 다음 상태 계산
    f_value = np.array(f(x0, u0)).flatten()
    x0_next = x0 + T * f_value
    t0_next = t0 + T
    return t0_next, x0_next, u0