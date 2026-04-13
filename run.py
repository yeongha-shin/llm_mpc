import numpy as np
import matplotlib.pyplot as plt
import time
from modules.mpc.sim_settings import *
from modules.mpc.kinematic_model import kinematic_model
from modules.mpc.acados_setting_left_right import acados_setting_left_right
from modules.mpc.acados_setting_follow_ship import acados_setting_follow_ship
from modules.mpc.shift import shift
from modules.mpc.get_closest_dist import get_closest_dist
from modules.bridge.preference_to_mpc import PreferenceToMPC

def draw_car(ax, x, y, psi, L, W, color):
    car_box = np.array([
        [-L/2, L/2,  L/2, -L/2, -L/2],
        [-W/2, -W/2, W/2,  W/2, -W/2]
    ])
    R = np.array([[np.cos(psi), -np.sin(psi)],
                  [np.sin(psi),  np.cos(psi)]])
    car_box = R @ car_box
    poly = plt.Polygon(
        list(zip(x + car_box[0], y + car_box[1])),
        color=color, alpha=0.8, ec='k', linewidth=1.2
    )
    ax.add_patch(poly)
    return poly

def main():
    # -------------------------------------------------------
    # 1. GP 모델 로드 + 초기 context로 파라미터 한번만 결정
    # -------------------------------------------------------
    bridge = PreferenceToMPC(model_path='gp_model.pkl')

    init_min_dist, _, _ = get_closest_dist(x0, obs_list, rob_rad)
    init_n_obstacles = obs_list.shape[0]
    init_context = [init_min_dist + rob_rad, init_n_obstacles]

    mpc_params = bridge.get_mpc_params(init_context)
    tc_cbf_gain_star = mpc_params['tc_cbf_gain']

    print(f"초기 context: dist={init_context[0]:.0f}m, n_obs={init_context[1]}")
    print(f"학습된 tc_cbf_gain: {tc_cbf_gain_star:.4f}")

    # -------------------------------------------------------
    # 2. solver 생성
    # -------------------------------------------------------
    n_solvers = 3
    solvers = {}
    for i in range(1, 3):
        solvers[i] = acados_setting_left_right(i)
    solvers[3] = acados_setting_follow_ship(3)

    # kinematic 모델
    _, _, _, f = kinematic_model()

    # -------------------------------------------------------
    # 3. 시뮬레이션 초기화
    # -------------------------------------------------------
    t0 = 0.0
    x0_sim = x0.copy()
    u0 = np.array([0.0, 0.0])
    nx = len(x0_sim)
    nu = len(u0)

    xx = x0_sim.reshape(-1, 1)
    uu = u0.reshape(-1, 1)
    t_hist = [t0]
    comptime = []
    min_dist_save = []

    # MPC 초기화
    for i in range(1, n_solvers + 1):
        solvers[i].constraints_set(0, 'lbx', x0_sim)
        solvers[i].constraints_set(0, 'ubx', x0_sim)
        for k in range(N+1):
            solvers[i].set(k, 'x', x0_sim)
        for k in range(N):
            solvers[i].set(k, 'u', np.zeros(nu))
            solvers[i].set(k, 'pi', np.zeros(nx))

    # -------------------------------------------------------
    # 4. 시각화 초기화
    # -------------------------------------------------------
    fig = plt.figure(figsize=(20, 8))
    ax_main = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax_v    = plt.subplot2grid((2, 4), (1, 0))
    ax_r    = plt.subplot2grid((2, 4), (1, 1))
    ax_comp = plt.subplot2grid((2, 4), (1, 2))
    ax_dist = plt.subplot2grid((2, 4), (1, 3))

    ax_main.set_xlabel('x [m]')
    ax_main.set_ylabel('y [m]')
    ax_main.grid(True)
    ax_main.set_aspect('equal')
    ax_main.plot(xs[0], xs[1], 'rx', markersize=10, linewidth=2)
    plt.ion()
    plt.show()

    # -------------------------------------------------------
    # 5. MPC 메인 루프
    # -------------------------------------------------------
    mpciter = 0
    obs_list_sim = obs_list.copy()
    topo_list = [1, -1, 0]
    costs = np.zeros(n_solvers)

    while (x0_sim[0] - xs[0]) < -0.1 and mpciter < sim_tim / T:

        # 최단 거리 계산 (시각화용)
        min_dist, _, _ = get_closest_dist(x0_sim, obs_list_sim, rob_rad)
        min_dist_save.append(min_dist)

        # Reference 궤적 생성
        yref_plot = []
        psid = np.arctan2(xs[1], xs[0])
        for j in range(N+1):
            xd = min(x0_sim[0] + vd * np.cos(psid) * T * j, xs[0])
            yref = np.array([xd, 0.0, psid, vd, 0.0, 0.0, 0.0])
            yref_plot.append(yref)

            for s in range(1, n_solvers+1):
                if j == N:
                    solvers[s].cost_set(N, 'yref', yref[:5])
                elif j == 0:
                    solvers[s].cost_set(N, 'yref', yref[:5])
                else:
                    solvers[s].cost_set(j, 'yref', yref)

        yref_plot = np.array(yref_plot)

        # 장애물 위치 업데이트
        obs_list_sim[:, 0:2] += obs_list_sim[:, 3:5] * T

        # solver 파라미터 설정 (tc_cbf_gain_star 고정)
        tic = time.time()

        for s in range(1, n_solvers+1):
            topo = topo_list[s-1]
            for j in range(N+1):
                if s <= 2:
                    p_val = np.array([
                        tc_cbf_gain_star,   # 학습된 고정값
                        15.0,               # tc_cbf_cparam
                        obs_list_sim[0, 0], obs_list_sim[0, 1],
                        obs_list_sim[0, 2], 0.0, 0.0,
                        float(topo)
                    ])
                else:
                    p_val = np.array([
                        0.03,               # ed_cbf_gain
                        obs_list_sim[0, 0], obs_list_sim[0, 1],
                        obs_list_sim[0, 2], 0.0, 0.0
                    ])
                solvers[s].set(j, 'p', p_val)

            solvers[s].solve()
            costs[s-1] = solvers[s].get_cost()

        comptime.append(time.time() - tic)

        # 최적 solver 선택
        d = np.argmin(costs) + 1
        d = 1  # 강제 선택

        u0 = solvers[d].get(0, 'u')
        mpc_pred_traj_opt = np.array([solvers[d].get(k, 'x') for k in range(N+1)]).T
        mpc_input_opt     = np.array([solvers[d].get(k, 'u') for k in range(N)]).T

        # Warm start
        for s in range(1, n_solvers+1):
            if solvers[s].get_status() == 4:
                for k in range(N+1):
                    solvers[s].set(k, 'x', mpc_pred_traj_opt[:, k])
                for k in range(N):
                    solvers[s].set(k, 'u', np.zeros(nu))
                    solvers[s].set(k, 'pi', np.zeros(nx))

        # 상태 업데이트
        t0, x0_sim, u0 = shift(T, t0, x0_sim, u0, f)

        for s in range(1, n_solvers + 1):
            solvers[s].constraints_set(0, 'lbx', x0_sim)
            solvers[s].constraints_set(0, 'ubx', x0_sim)

        t_hist.append(t0)
        xx = np.hstack([xx, x0_sim.reshape(-1, 1)])
        uu = np.hstack([uu, u0.reshape(-1, 1)])

        mpciter += 1

        # 시각화 (5스텝마다)
        if mpciter % 5 == 0:
            ax_main.cla()
            ax_main.grid(True)
            ax_main.set_aspect('equal')
            ax_main.plot(xs[0], xs[1], 'rx', markersize=10, linewidth=2)

            ax_main.plot(xx[0,:], xx[1,:], 'b-', linewidth=2)

            for s in range(1, n_solvers+1):
                tmp = np.array([solvers[s].get(k, 'x') for k in range(N+1)]).T
                ax_main.plot(tmp[0,:], tmp[1,:], 'c-', linewidth=1.5)
            ax_main.plot(mpc_pred_traj_opt[0,:], mpc_pred_traj_opt[1,:], 'm-', linewidth=2.5)

            ax_main.plot(yref_plot[:,0], yref_plot[:,1], 'r.', linewidth=1.5)

            pos = xx[:2, -1]
            ang = np.linspace(0, 2*np.pi, 30)
            ax_main.plot(pos[0] + rob_rad*np.cos(ang),
                        pos[1] + rob_rad*np.sin(ang), 'k-', linewidth=1.5)
            draw_car(ax_main, x0_sim[0], x0_sim[1], x0_sim[2], 4, 1.5, [0.2, 0.6, 1.0])
            ax_main.quiver(x0_sim[0], x0_sim[1],
                          np.cos(x0_sim[2]), np.sin(x0_sim[2]),
                          scale=5, color='k', linewidth=2)

            for ob in range(obs_list_sim.shape[0]):
                ox  = obs_list_sim[ob, 0]
                oy  = obs_list_sim[ob, 1]
                or_ = obs_list_sim[ob, 2]
                circle = plt.Circle((ox, oy), or_, color=[0.8, 0.2, 0.2], alpha=0.5)
                ax_main.add_patch(circle)

            ax_main.set_xlim([-5, xs[0]+5])
            ax_main.set_ylim([-150, 150])
            ax_main.set_title(f'Time: {t_hist[-1]:.2f} s | '
                              f'tc_cbf_gain: {tc_cbf_gain_star:.4f} | '
                              f'dist: {init_context[0]:.0f}m, n_obs: {init_context[1]}')
            ax_main.set_xlabel('x [m]')
            ax_main.set_ylabel('y [m]')

            t_arr = np.array(t_hist)

            ax_v.cla(); ax_v.grid(True)
            ax_v.plot(t_arr, xx[3,:], 'k-', linewidth=1.5)
            ax_v.set_ylabel('v [m/s]'); ax_v.set_xlabel('time [s]')

            ax_r.cla(); ax_r.grid(True)
            ax_r.plot(t_arr, np.rad2deg(xx[4,:]), 'k-', linewidth=1.5)
            ax_r.set_ylabel('ω [deg/s]'); ax_r.set_xlabel('time [s]')

            ax_comp.cla(); ax_comp.grid(True)
            ax_comp.plot(t_arr[1:], np.array(comptime)*1e3/n_solvers, 'k-', linewidth=1.5)
            ax_comp.set_ylabel('comp. time [ms]'); ax_comp.set_xlabel('time [s]')

            ax_dist.cla(); ax_dist.grid(True)
            ax_dist.plot(t_arr[1:], np.array(min_dist_save), 'k-', linewidth=1.5)
            ax_dist.plot(t_arr[1:], np.zeros(len(min_dist_save)), 'r--', linewidth=1.5)
            ax_dist.set_ylabel('Closest distance [m]'); ax_dist.set_xlabel('time [s]')

            plt.tight_layout()
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    print('Simulation finished!')
    print(f'사용된 tc_cbf_gain: {tc_cbf_gain_star:.4f}')

if __name__ == '__main__':
    main()