import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import *
from sim_settings import *
from kinematic_model import kinematic_model
from scipy.linalg import block_diag

def acados_setting_left_right(ind):
    # 모델 생성
    states, controls, f_expl, f = kinematic_model()

    model = AcadosModel()
    model.name = f'kinematic{ind}'
    model.x = states
    model.u = controls
    model.f_expl_expr = f_expl

    nx = states.shape[0]
    nu = controls.shape[0]

    # 비용 가중치
    Q = np.diag([0.001, 0.05, 50, 300, 200])
    QN = Q * 5
    Rd = np.diag([500, 100])

    # OCP 정의
    ocp = AcadosOcp()
    ocp.model = model

    # 파라미터 벡터: [tc_cbf_gain; tc_cbf_cparam; 장애물 정보 * N_obs]
    # 각 장애물당 6개: [x, y, r, xd, yd, topo]
    p = SX.sym('p', 2 + 6 * N_obs)
    ocp.model.p = p
    ocp.parameter_values = np.zeros(p.shape[0])

    # 비용 함수
    # 초기 스텝
    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.cost.W_0 = Rd
    ocp.cost.yref_0 = np.zeros(nu)
    ocp.model.cost_y_expr_0 = controls

    # 경로 스텝
    ny = nx + nu
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = block_diag(Q, Rd)
    ocp.cost.yref = np.zeros(ny)
    ocp.model.cost_y_expr = vertcat(states, controls)

    # 터미널 스텝
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W_e = QN
    ocp.cost.yref_e = np.zeros(nx)
    ocp.model.cost_y_expr_e = states

    # TC-CBF 제약
    h_expr = []
    tc_cbf_gain_p = p[0]
    tc_cbf_cparam_p = p[1]

    for j in range(N_obs):
        t_x  = p[2 + 6*j]
        t_y  = p[3 + 6*j]
        t_r  = p[4 + 6*j]
        t_xd = p[5 + 6*j]
        t_yd = p[6 + 6*j]
        t_t  = p[7 + 6*j]

        t_xn = t_x + t_xd * T
        t_yn = t_y + t_yd * T

        o_x   = model.x[0]
        o_y   = model.x[1]
        o_psi = model.x[2]
        o_v   = model.x[3]

        o_xd  = model.f_expl_expr[0] * T
        o_yd  = model.f_expl_expr[1] * T
        o_psid = model.f_expl_expr[2] * T
        o_xn  = o_x + o_xd
        o_yn  = o_y + o_yd
        o_psin = o_psi + o_psid

        R = model.x[3] / r_max * tc_cbf_cparam_p

        hk = sqrt((o_x + R * cos(o_psi - pi/2 * t_t) - t_x)**2 +
                  (o_y + R * sin(o_psi - pi/2 * t_t) - t_y)**2) - (rob_rad + t_r + R)

        hkn = sqrt((o_xn + R * cos(o_psin - pi/2 * t_t) - t_xn)**2 +
                   (o_yn + R * sin(o_psin - pi/2 * t_t) - t_yn)**2) - (rob_rad + t_r + R)

        g = hkn - (1 - tc_cbf_gain_p) * hk
        h_expr.append(g)

    h_expr = vertcat(*h_expr)
    ocp.model.con_h_expr = h_expr

    # 제약조건
    ocp.constraints.x0 = np.array([0, 0, 0, 0, 0])

    ocp.constraints.lbu = np.array([-acc_max, -rd_max])
    ocp.constraints.ubu = np.array([ acc_max,  rd_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([v_min, r_min])
    ocp.constraints.ubx = np.array([v_max, r_max])
    ocp.constraints.idxbx = np.array([3, 4])

    ocp.constraints.lbx_e = np.array([v_min, r_min])
    ocp.constraints.ubx_e = np.array([v_max, r_max])
    ocp.constraints.idxbx_e = np.array([3, 4])

    ocp.constraints.uh = 1e8 * np.ones(N_obs)
    ocp.constraints.lh = -1e-10 * np.ones(N_obs)

    # Soft constraint
    ocp.constraints.idxsh = np.arange(N_obs)
    Zh = 1e6 * np.ones(N_obs)
    zh = np.zeros(N_obs)
    ocp.cost.Zl = Zh
    ocp.cost.Zu = Zh
    ocp.cost.zl = zh
    ocp.cost.zu = zh

    # Solver 옵션
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T * N
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_mu0 = 1e3
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.qp_solver_iter_max = 20

    ocp_solver = AcadosOcpSolver(ocp)
    return ocp_solver