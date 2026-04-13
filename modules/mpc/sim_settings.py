import numpy as np

T = 1
N = 30
N_obs = 1

rob_rad = 15
rob_safe_bumper = 0.01
v_max = 5
v_min = 0
r_max = np.pi / 6
r_min = -np.pi / 6
acc_max = 0.5
rd_max = 0.5

tc_cbf_gain = 0.015
tc_cbf_cparam = 15
ed_cbf_gain = 0.03

# tc_cbf_gain = 0.015   # 낮을수록 보수적(일찍 회피), 높을수록 늦게 회피
# tc_cbf_cparam = 15    # 회피 곡선 반경 → 클수록 크게 돌아서 회피
# ed_cbf_gain = 0.03    # follow ship 모드 보수성

sim_tim = 400
vd = 3

x0 = np.array([0, 0, 0, vd, 0])
xs = np.array([1200, 0, 0, vd, 0])

obs_list = np.zeros((1, 5))
obs_list[0] = [400, 2, 35, 1, 0.0]