import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit


class GPBPL:
    def __init__(self):
        # theta 범위
        self.theta_min = 0.001
        self.theta_max = 0.05

        # GP 하이퍼파라미터
        self.length_scale = 0.015
        self.signal_var = 1.0
        self.noise_var = 1e-4

        # 사전 평균
        self.prior_mean = 0.0

        # 관측 데이터
        self.observations = []  # (theta_A, theta_B, preference)

        # 학습 포인트 (고정 그리드)
        self.n_points = 50
        self.X = np.linspace(self.theta_min, self.theta_max, self.n_points)

        # Laplace approximation 결과 캐시
        self.f_map = None
        self.K = None
        self._build_kernel_matrix()

    def _build_kernel_matrix(self):
        n = len(self.X)
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self._kernel(self.X[i], self.X[j])
        self.K += self.noise_var * np.eye(n)

    def _kernel(self, x1, x2):
        return self.signal_var * np.exp(
            -0.5 * ((x1 - x2) / self.length_scale) ** 2
        )

    def _sigmoid(self, x):
        return expit(x)

    def _get_f_values(self, f, theta):
        """theta에 해당하는 f값 선형 보간"""
        idx = np.searchsorted(self.X, theta)
        idx = np.clip(idx, 0, len(self.X) - 1)
        return f[idx]

    def _log_likelihood(self, f):
        """pairwise 비교 관측에 대한 log likelihood"""
        log_lik = 0.0
        for theta_A, theta_B, pref in self.observations:
            f_A = self._get_f_values(f, theta_A)
            f_B = self._get_f_values(f, theta_B)
            diff = f_A - f_B
            if pref == 1:
                log_lik += np.log(self._sigmoid(diff) + 1e-10)
            else:
                log_lik += np.log(self._sigmoid(-diff) + 1e-10)
        return log_lik

    def _log_likelihood_grad(self, f):
        """log likelihood의 gradient"""
        grad = np.zeros(len(f))
        for theta_A, theta_B, pref in self.observations:
            idx_A = np.searchsorted(self.X, theta_A)
            idx_B = np.searchsorted(self.X, theta_B)
            idx_A = np.clip(idx_A, 0, len(self.X) - 1)
            idx_B = np.clip(idx_B, 0, len(self.X) - 1)

            f_A = f[idx_A]
            f_B = f[idx_B]
            diff = f_A - f_B

            if pref == 1:
                s = self._sigmoid(-diff)
            else:
                s = -self._sigmoid(diff)

            grad[idx_A] += s
            grad[idx_B] -= s
        return grad

    def _log_likelihood_hessian_diag(self, f):
        """log likelihood의 Hessian 대각 원소"""
        hess_diag = np.zeros(len(f))
        for theta_A, theta_B, pref in self.observations:
            idx_A = np.searchsorted(self.X, theta_A)
            idx_B = np.searchsorted(self.X, theta_B)
            idx_A = np.clip(idx_A, 0, len(self.X) - 1)
            idx_B = np.clip(idx_B, 0, len(self.X) - 1)

            f_A = f[idx_A]
            f_B = f[idx_B]
            diff = f_A - f_B

            s = self._sigmoid(diff) * self._sigmoid(-diff)  # σ(1-σ)

            hess_diag[idx_A] -= s
            hess_diag[idx_B] -= s
        return hess_diag

    def _laplace_approximation(self):
        """MAP 추정 + Hessian으로 posterior 근사"""
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

        # Hessian: W = -∇∇ log p(y|f)
        W_diag = -self._log_likelihood_hessian_diag(f_map)
        W_diag = np.maximum(W_diag, 1e-10)  # 수치 안정성

        return f_map, W_diag

    def posterior(self, theta_test):
        """posterior mean과 variance 계산"""
        n = len(self.X)

        if len(self.observations) == 0:
            # 관측 없으면 사전분포
            mu = np.ones(len(theta_test)) * self.prior_mean
            var = np.array([self._kernel(t, t) for t in theta_test])
            return mu, var

        # Laplace approximation
        f_map, W_diag = self._laplace_approximation()
        self.f_map = f_map

        cho_K = cho_factor(self.K)

        # W^{1/2}
        W_sqrt = np.sqrt(W_diag)

        # B = I + W^{1/2} K W^{1/2}
        B = np.eye(n) + np.diag(W_sqrt) @ self.K @ np.diag(W_sqrt)
        cho_B = cho_factor(B)

        # 테스트 포인트에 대한 k_star
        k_star = np.array([[self._kernel(t, x) for x in self.X]
                           for t in theta_test])

        # Posterior mean
        mu = self.prior_mean + k_star @ cho_solve(cho_K, f_map - self.prior_mean)

        # Posterior variance
        var = np.zeros(len(theta_test))
        for i, t in enumerate(theta_test):
            k_tt = self._kernel(t, t)
            v = cho_solve(cho_B, np.diag(W_sqrt) @ k_star[i])
            var[i] = k_tt - k_star[i] @ np.diag(W_sqrt) @ v
        var = np.maximum(var, 1e-10)

        return mu, var

    def add_observation(self, theta_A, theta_B, preference):
        self.observations.append((theta_A, theta_B, preference))

    def get_current_best(self):
        theta_range = self.get_theta_range()
        mu, _ = self.posterior(theta_range)
        return theta_range[np.argmax(mu)]

    def get_theta_range(self, n_points=100):
        return np.linspace(self.theta_min, self.theta_max, n_points)

    def next_query(self):
        theta_range = self.get_theta_range()
        mu, var = self.posterior(theta_range)

        # uncertainty 가장 높은 포인트
        theta_A = theta_range[np.argmax(var)]

        # 현재 best (posterior mean 최대)
        theta_B = theta_range[np.argmax(mu)]

        return theta_A, theta_B