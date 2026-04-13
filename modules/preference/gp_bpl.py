import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit


class GPBPL:
    def __init__(self):
        # theta 범위
        self.theta_min = 0.001
        self.theta_max = 0.05

        # context 범위
        self.min_dist_min = 500
        self.min_dist_max = 2000
        self.n_obs_min = 1
        self.n_obs_max = 5

        # GP 하이퍼파라미터
        self.length_scale = 0.3  # 정규화된 공간에서의 length scale
        self.signal_var = 1.0
        self.noise_var = 1e-4

        # 사전 평균
        self.prior_mean = 0.0

        # 관측 데이터
        # (theta_A, context_A, theta_B, context_B, preference)
        self.observations = []

        # 학습 포인트 그리드 (3차원)
        self.n_theta = 10
        self.n_dist = 5
        self.n_obs = 5
        self.X = self._build_grid()

        # 커널 행렬
        self.K = None
        self._build_kernel_matrix()

    def _build_grid(self):
        """3차원 그리드 생성"""
        theta_grid = np.linspace(0, 1, self.n_theta)
        dist_grid = np.linspace(0, 1, self.n_dist)
        obs_grid = np.linspace(0, 1, self.n_obs)

        points = []
        for t in theta_grid:
            for d in dist_grid:
                for o in obs_grid:
                    points.append([t, d, o])
        return np.array(points)

    def _normalize(self, theta, min_dist, n_obstacles):
        """입력값 정규화 (0~1)"""
        t_norm = (theta - self.theta_min) / (self.theta_max - self.theta_min)
        d_norm = (min_dist - self.min_dist_min) / (self.min_dist_max - self.min_dist_min)
        o_norm = (n_obstacles - self.n_obs_min) / (self.n_obs_max - self.n_obs_min)
        return np.array([t_norm, d_norm, o_norm])

    def _denormalize_theta(self, t_norm):
        """theta 역정규화"""
        return t_norm * (self.theta_max - self.theta_min) + self.theta_min

    def _kernel(self, x1, x2):
        """RBF 커널 (3차원)"""
        diff = x1 - x2
        return self.signal_var * np.exp(
            -0.5 * np.dot(diff, diff) / self.length_scale ** 2
        )

    def _build_kernel_matrix(self):
        n = len(self.X)
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self._kernel(self.X[i], self.X[j])
        self.K += self.noise_var * np.eye(n)

    def _sigmoid(self, x):
        return expit(x)

    def _get_f_value(self, f, x_norm):
        """정규화된 입력에 해당하는 f값 (최근접 포인트)"""
        dists = np.linalg.norm(self.X - x_norm, axis=1)
        idx = np.argmin(dists)
        return f[idx], idx

    def _log_likelihood(self, f):
        log_lik = 0.0
        for theta_A, ctx_A, theta_B, ctx_B, pref in self.observations:
            x_A = self._normalize(theta_A, ctx_A[0], ctx_A[1])
            x_B = self._normalize(theta_B, ctx_B[0], ctx_B[1])

            f_A, _ = self._get_f_value(f, x_A)
            f_B, _ = self._get_f_value(f, x_B)

            diff = f_A - f_B
            if pref == 1:
                log_lik += np.log(self._sigmoid(diff) + 1e-10)
            else:
                log_lik += np.log(self._sigmoid(-diff) + 1e-10)
        return log_lik

    def _log_likelihood_grad(self, f):
        grad = np.zeros(len(f))
        for theta_A, ctx_A, theta_B, ctx_B, pref in self.observations:
            x_A = self._normalize(theta_A, ctx_A[0], ctx_A[1])
            x_B = self._normalize(theta_B, ctx_B[0], ctx_B[1])

            f_A, idx_A = self._get_f_value(f, x_A)
            f_B, idx_B = self._get_f_value(f, x_B)

            diff = f_A - f_B
            if pref == 1:
                s = self._sigmoid(-diff)
            else:
                s = -self._sigmoid(diff)

            grad[idx_A] += s
            grad[idx_B] -= s
        return grad

    def _log_likelihood_hessian_diag(self, f):
        hess_diag = np.zeros(len(f))
        for theta_A, ctx_A, theta_B, ctx_B, pref in self.observations:
            x_A = self._normalize(theta_A, ctx_A[0], ctx_A[1])
            x_B = self._normalize(theta_B, ctx_B[0], ctx_B[1])

            f_A, idx_A = self._get_f_value(f, x_A)
            f_B, idx_B = self._get_f_value(f, x_B)

            diff = f_A - f_B
            s = self._sigmoid(diff) * self._sigmoid(-diff)

            hess_diag[idx_A] -= s
            hess_diag[idx_B] -= s
        return hess_diag

    def _laplace_approximation(self):
        n = len(self.X)
        f_init = np.zeros(n)
        cho = cho_factor(self.K)

        def neg_log_posterior(f):
            log_lik = self._log_likelihood(f)
            diff = f - self.prior_mean
            log_prior = -0.5 * diff @ cho_solve(cho, diff)
            return -(log_lik + log_prior)

        def neg_log_posterior_grad(f):
            grad_lik = self._log_likelihood_grad(f)
            diff = f - self.prior_mean
            grad_prior = -cho_solve(cho, diff)
            return -(grad_lik + grad_prior)

        result = minimize(
            neg_log_posterior,
            f_init,
            jac=neg_log_posterior_grad,
            method='L-BFGS-B'
        )

        f_map = result.x
        W_diag = -self._log_likelihood_hessian_diag(f_map)
        W_diag = np.maximum(W_diag, 1e-10)

        return f_map, W_diag

    def posterior(self, X_test):
        """
        X_test: (n, 3) 정규화된 입력 배열
        """
        n = len(self.X)

        if len(self.observations) == 0:
            mu = np.ones(len(X_test)) * self.prior_mean
            var = np.array([self._kernel(x, x) for x in X_test])
            return mu, var

        f_map, W_diag = self._laplace_approximation()
        cho_K = cho_factor(self.K)

        W_sqrt = np.sqrt(W_diag)
        B = np.eye(n) + np.diag(W_sqrt) @ self.K @ np.diag(W_sqrt)
        cho_B = cho_factor(B)

        k_star = np.array([[self._kernel(t, x) for x in self.X]
                           for t in X_test])

        mu = self.prior_mean + k_star @ cho_solve(cho_K,
                                                  f_map - self.prior_mean)

        var = np.zeros(len(X_test))
        for i, t in enumerate(X_test):
            k_tt = self._kernel(t, t)
            v = cho_solve(cho_B, np.diag(W_sqrt) @ k_star[i])
            var[i] = k_tt - k_star[i] @ np.diag(W_sqrt) @ v
        var = np.maximum(var, 1e-10)

        return mu, var

    def add_observation(self, theta_A, context_A, theta_B, context_B, preference):
        """
        context_A, context_B: [min_dist, n_obstacles]
        preference: 1이면 A 선호, 0이면 B 선호
        """
        self.observations.append(
            (theta_A, context_A, theta_B, context_B, preference)
        )

    def get_current_best(self, context):
        """
        주어진 context에서 가장 선호되는 theta 반환
        context: [min_dist, n_obstacles]
        """
        theta_range = np.linspace(self.theta_min, self.theta_max, 50)
        X_test = np.array([
            self._normalize(t, context[0], context[1])
            for t in theta_range
        ])
        mu, _ = self.posterior(X_test)
        best_theta = theta_range[np.argmax(mu)]
        return best_theta

    def next_query(self, context):
        """
        주어진 context에서 다음 질문할 theta 쌍 반환
        context: [min_dist, n_obstacles]
        """
        theta_range = np.linspace(self.theta_min, self.theta_max, 50)
        X_test = np.array([
            self._normalize(t, context[0], context[1])
            for t in theta_range
        ])
        mu, var = self.posterior(X_test)

        # uncertainty 가장 높은 포인트 (exploration)
        theta_A = theta_range[np.argmax(var)]

        # 현재 best (exploitation)
        theta_B = theta_range[np.argmax(mu)]

        return theta_A, theta_B

    def get_theta_range(self, n_points=50):
        return np.linspace(self.theta_min, self.theta_max, n_points)