import numpy as np
import pickle
from modules.preference.gp_bpl import GPBPL


class PreferenceToMPC:
    def __init__(self, model_path='gp_model.pkl'):
        with open(model_path, 'rb') as f:
            self.gp = pickle.load(f)
        print(f"GP 모델 로드 완료: {model_path}")

    def get_mpc_params(self, context):
        """
        context: [min_dist, n_obstacles]
        """
        tc_cbf_gain = self.gp.get_current_best(context)

        return {
            'tc_cbf_gain': tc_cbf_gain,
        }

    def apply_to_solver(self, solvers, obs_list, N, topo_list, context):
        """
        현재 context 기반으로 파라미터 계산 후 solver에 적용
        """
        mpc_params = self.get_mpc_params(context)
        tc_cbf_gain = mpc_params['tc_cbf_gain']

        print(f"현재 context: dist={context[0]}m, n_obs={context[1]} "
              f"→ tc_cbf_gain={tc_cbf_gain:.4f}")

        for s in range(1, len(solvers) + 1):
            topo = topo_list[s - 1]
            for j in range(N + 1):
                if s <= 2:
                    p_val = np.array([
                        tc_cbf_gain,
                        15.0,
                        obs_list[0, 0], obs_list[0, 1],
                        obs_list[0, 2], 0.0, 0.0,
                        float(topo)
                    ])
                else:
                    p_val = np.array([
                        0.03,
                        obs_list[0, 0], obs_list[0, 1],
                        obs_list[0, 2], 0.0, 0.0
                    ])
                solvers[s].set(j, 'p', p_val)