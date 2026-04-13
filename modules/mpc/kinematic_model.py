from casadi import *

def kinematic_model():
    # 상태/입력 심볼릭 변수 정의
    x     = SX.sym('x')
    y     = SX.sym('y')
    theta = SX.sym('theta')
    v     = SX.sym('v')
    r     = SX.sym('r')
    states = vertcat(x, y, theta, v, r)
    nx = states.shape[0]

    vd = SX.sym('vd')
    rd = SX.sym('rd')
    controls = vertcat(vd, rd)
    nu = controls.shape[0]

    # 연속시간 동역학
    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        r,
        vd,
        rd
    )

    # CasADi Function
    f = Function('f', [states, controls], [f_expl])

    return states, controls, f_expl, f