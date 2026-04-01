"""
GP-BPL + MPC 운항 스타일 학습 테스트 (수정판)
- variance가 iteration마다 단조 감소하도록 수식 수정
- utility 추정의 일관성 개선
- 수렴 기준 도달 가능
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(42)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
MAX_ITER     = 10
ENT_THRESH   = 0.25
THETA_NAMES  = ["q_pos", "q_vel", "r_n", "S"]
THETA_BOUNDS = np.array([
    [0.5, 5.0],
    [0.1, 3.0],
    [0.1, 3.0],
    [1.0, 10.0],
])
TRUE_THETA = np.array([3.5, 1.2, 2.0, 7.5])

# ──────────────────────────────────────────
# 더미 운항자 응답 풀
# TRUE_THETA에 일관된 방향의 응답으로 정리
# ──────────────────────────────────────────
DUMMY_RESPONSES = [
    {
        "text": "A 경로가 더 안전합니다. 상대선과 충분한 거리를 유지하고 싶습니다.",
        "parsed": {
            "q_pos": {"direction": 0.85, "confidence": 0.90, "evidence": "안전거리 직접 언급"},
            "q_vel": {"direction": 0.50, "confidence": 0.00, "evidence": None},
            "r_n":   {"direction": 0.50, "confidence": 0.00, "evidence": None},
            "S":     {"direction": 0.80, "confidence": 0.60, "evidence": "안전 강조 간접 추론"},
        },
    },
    {
        "text": "COLREG 규정은 항상 지켜야 합니다. 임무보다 안전이 우선입니다.",
        "parsed": {
            "q_pos": {"direction": 0.75, "confidence": 0.70, "evidence": "안전 우선 언급"},
            "q_vel": {"direction": 0.35, "confidence": 0.55, "evidence": "임무 < 안전"},
            "r_n":   {"direction": 0.50, "confidence": 0.00, "evidence": None},
            "S":     {"direction": 0.90, "confidence": 0.92, "evidence": "COLREG 직접 언급"},
        },
    },
    {
        "text": "속도 변화가 적은 경로가 좋습니다. 급격한 조타는 선체에 좋지 않습니다.",
        "parsed": {
            "q_pos": {"direction": 0.65, "confidence": 0.45, "evidence": "안정 운항 간접"},
            "q_vel": {"direction": 0.50, "confidence": 0.00, "evidence": None},
            "r_n":   {"direction": 0.85, "confidence": 0.88, "evidence": "속도 변화 억제 직접"},
            "S":     {"direction": 0.70, "confidence": 0.40, "evidence": "안정 운항 간접"},
        },
    },
    {
        "text": "야간이라 더 넓게 피하겠습니다. 안전 마진을 충분히 두고 싶습니다.",
        "parsed": {
            "q_pos": {"direction": 0.88, "confidence": 0.88, "evidence": "넓게 피하기 직접"},
            "q_vel": {"direction": 0.40, "confidence": 0.40, "evidence": "여유 운항 간접"},
            "r_n":   {"direction": 0.65, "confidence": 0.45, "evidence": "여유 운항 간접"},
            "S":     {"direction": 0.80, "confidence": 0.65, "evidence": "안전 강조 추론"},
        },
    },
    {
        "text": "우현 통항을 유지하겠습니다. COLREG 규정대로 해야죠.",
        "parsed": {
            "q_pos": {"direction": 0.70, "confidence": 0.55, "evidence": "안전 운항 간접"},
            "q_vel": {"direction": 0.50, "confidence": 0.00, "evidence": None},
            "r_n":   {"direction": 0.60, "confidence": 0.35, "evidence": "안정 운항 간접"},
            "S":     {"direction": 0.92, "confidence": 0.95, "evidence": "COLREG 직접"},
        },
    },
    {
        "text": "급격한 변침보다 완만하게 피항하고 싶습니다.",
        "parsed": {
            "q_pos": {"direction": 0.72, "confidence": 0.60, "evidence": "피항 중시 간접"},
            "q_vel": {"direction": 0.45, "confidence": 0.35, "evidence": "급격함 회피 간접"},
            "r_n":   {"direction": 0.82, "confidence": 0.80, "evidence": "완만한 변침 직접"},
            "S":     {"direction": 0.75, "confidence": 0.55, "evidence": "규정 준수 간접"},
        },
    },
    {
        "text": "안전이 최우선입니다. 조금 돌아가도 괜찮습니다.",
        "parsed": {
            "q_pos": {"direction": 0.90, "confidence": 0.85, "evidence": "안전 최우선 직접"},
            "q_vel": {"direction": 0.30, "confidence": 0.70, "evidence": "시간 양보 직접"},
            "r_n":   {"direction": 0.65, "confidence": 0.40, "evidence": "안정 운항 간접"},
            "S":     {"direction": 0.85, "confidence": 0.70, "evidence": "안전 강조 추론"},
        },
    },
    {
        "text": "속도를 너무 자주 바꾸지 않는 경로가 좋습니다.",
        "parsed": {
            "q_pos": {"direction": 0.60, "confidence": 0.40, "evidence": "안정 간접"},
            "q_vel": {"direction": 0.45, "confidence": 0.50, "evidence": "속도 일정 간접"},
            "r_n":   {"direction": 0.88, "confidence": 0.85, "evidence": "속도 변화 억제 직접"},
            "S":     {"direction": 0.70, "confidence": 0.45, "evidence": "안전 간접"},
        },
    },
]


# ──────────────────────────────────────────
# GP-BPL (수정판)
# ──────────────────────────────────────────
class GPBPL:
    """
    핵심 수정:
      variance = 1 - K_sum / (noise + K_sum)
      → noise=1.0 기준으로 iteration마다 단조 감소 보장
    """

    def __init__(self, dim, bounds, length_scale=1.2, var_noise=1.0):
        self.dim       = dim
        self.bounds    = bounds
        self.ls        = length_scale
        self.var_noise = var_noise   # variance 계산용 noise 파라미터
        self.obs_pair  = []          # (xA_n, xB_n, chose_A)
        self.obs_rea   = []          # (param_idx, direction, confidence)

    def _norm(self, theta):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.array(theta) - lo) / (hi - lo + 1e-8)

    def _k(self, x1, x2):
        d = x1 - x2
        return float(np.exp(-0.5 * np.dot(d, d) / self.ls**2))

    def utility(self, theta):
        """
        U(θ) 추정:
          L1: 쌍비교 신호  — A 선택 → θA 근처 U 높임, θB 낮춤
          L2: reasoning 신호 — confidence 가중치로 θ 방향 반영
        """
        x = self._norm(theta)
        u = 0.0

        for xA, xB, chose_A in self.obs_pair:
            diff = self._k(x, xA) - self._k(x, xB)
            u += diff if chose_A else -diff

        for idx, direction, conf in self.obs_rea:
            target      = np.zeros(self.dim)
            target[idx] = direction
            u += conf * self._k(x, target) * 0.4

        return u

    def variance(self, theta):
        """
        posterior variance 근사
        수식: var = 1 - K_sum / (noise + K_sum)
          - 관측이 없으면 1.0 (완전 불확실)
          - 관측 근처일수록 K_sum 커져서 variance 감소
          - 새 관측 추가 시 단조 감소 보장
        """
        x = self._norm(theta)
        if not self.obs_pair:
            return 1.0

        all_x = ([xA for xA, _, _ in self.obs_pair] +
                 [xB for _, xB, _ in self.obs_pair])
        K_sum = sum(self._k(x, xi) ** 2 for xi in all_x)

        var = 1.0 - K_sum / (self.var_noise + K_sum)
        return float(max(0.01, var))

    def mean_variance(self, n=100):
        """전체 파라미터 공간의 평균 불확실성 — 수렴 지표"""
        pts    = np.random.uniform(0, 1, (n, self.dim))
        thetas = pts * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return float(np.mean([self.variance(t) for t in thetas]))

    def acquisition(self, n_cand=400):
        """Maximum Variance: 가장 불확실한 두 지점 선택"""
        cands = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n_cand, self.dim)
        )
        vars_ = [self.variance(c) for c in cands]

        iA = int(np.argmax(vars_))
        tA = cands[iA]

        # θB: variance 높고 θA와 충분히 먼 지점
        scores = []
        for i, c in enumerate(cands):
            if i == iA:
                scores.append(-np.inf)
                continue
            dist = np.linalg.norm(self._norm(c) - self._norm(tA))
            scores.append(vars_[i] * min(1.0, dist * 2.0))
        iB = int(np.argmax(scores))
        return tA, cands[iB]

    def update_pairwise(self, tA, tB, chose_A):
        self.obs_pair.append((self._norm(tA), self._norm(tB), chose_A))

    def update_reasoning(self, parsed):
        for i, name in enumerate(THETA_NAMES):
            info = parsed[name]
            if info["confidence"] > 0.0:
                self.obs_rea.append((i, info["direction"], info["confidence"]))

    def theta_star(self, n_cand=800):
        """utility 최댓값 위치 = 추정된 운항 선호 θ*"""
        cands = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n_cand, self.dim)
        )
        utils = [self.utility(c) for c in cands]
        return cands[int(np.argmax(utils))]

    def param_variances(self):
        """각 파라미터 축별 평균 분산 (시각화용)"""
        result = []
        for i in range(self.dim):
            samps = np.linspace(self.bounds[i, 0], self.bounds[i, 1], 30)
            v_list = []
            for s in samps:
                t    = TRUE_THETA.copy()
                t[i] = s
                v_list.append(self.variance(t))
            result.append(float(np.mean(v_list)))
        return result


# ──────────────────────────────────────────
# MPC 경로 시뮬레이터
# ──────────────────────────────────────────
class MPCSim:
    def __init__(self):
        self.start   = np.array([0.0,    0.0])
        self.goal    = np.array([1000.0, 0.0])
        self.vessels = [
            {"pos": np.array([380.0, -70.0]), "r": 80},
            {"pos": np.array([650.0,  90.0]), "r": 65},
        ]

    def generate_path(self, theta, n=40):
        q_pos, q_vel, r_n, S = theta
        rng  = np.random.default_rng(int(abs(sum(theta * 97))) % 9999)
        path = [self.start.copy()]

        for ti in np.linspace(0, 1, n)[1:]:
            x = ti * self.goal[0]
            y = 0.0

            for v in self.vessels:
                vx, vy = v["pos"]
                df    = np.exp(-((x - vx) ** 2) / (160.0 ** 2))
                avoid = (q_pos / 5.0) * 1.3 * (v["r"] / 65.0) * df
                y    -= vy * avoid

            y += (S / 10.0) * 28 * np.sin(np.pi * ti)

            if len(path) > 1:
                alpha = min(0.88, r_n / 3.5)
                y     = alpha * path[-1][1] + (1 - alpha) * y

            noise_scale = max(0.0, 2.0 - q_vel) * 3.0
            y += rng.normal(0, noise_scale) * (1.0 - min(0.9, r_n / 4.0))

            path.append(np.array([x, y]))

        return np.array(path)

    def metrics(self, path):
        min_d = min(
            np.linalg.norm(p - v["pos"]) - v["r"]
            for p in path for v in self.vessels
        )
        diffs  = np.diff(path[:, 1])
        smooth = 1.0 / (1.0 + float(np.std(diffs)))
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        return {
            "min_clearance_m": round(float(min_d), 1),
            "smoothness":      round(smooth, 3),
            "path_length_m":   round(length, 1),
        }


# ──────────────────────────────────────────
# 더미 LLM
# ──────────────────────────────────────────
def llm_explain(theta_A, theta_B, m_A, m_B):
    lines = []
    dc = m_A["min_clearance_m"] - m_B["min_clearance_m"]
    if dc > 15:
        lines.append("Path A maintains wider clearance from vessels.")
    elif dc < -15:
        lines.append("Path B maintains wider clearance from vessels.")
    else:
        lines.append("Both paths have similar vessel clearance.")

    dl = m_A["path_length_m"] - m_B["path_length_m"]
    if dl > 40:
        lines.append("Path A is longer due to wider detour.")
    elif dl < -40:
        lines.append("Path B is longer due to wider detour.")

    ds = m_A["smoothness"] - m_B["smoothness"]
    if ds > 0.03:
        lines.append("Path A has smoother heading changes.")
    elif ds < -0.03:
        lines.append("Path B has smoother heading changes.")

    return " ".join(lines)


def llm_parse(response_dict):
    return response_dict["parsed"]


def dummy_operator(theta_A, theta_B, iteration):
    dist_A  = np.linalg.norm(theta_A - TRUE_THETA)
    dist_B  = np.linalg.norm(theta_B - TRUE_THETA)
    chose_A = dist_A < dist_B
    resp    = DUMMY_RESPONSES[iteration % len(DUMMY_RESPONSES)]
    return chose_A, resp


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────
COLORS = ["#534AB7", "#1D9E75", "#EF9F27", "#D85A30"]


def draw_vessels(ax, mpc):
    for v in mpc.vessels:
        c = plt.Circle(v["pos"], v["r"], color="#E24B4A", alpha=0.2, zorder=2)
        ax.add_patch(c)
        ax.plot(*v["pos"], "x", color="#A32D2D", ms=8, zorder=4)


def plot_iter_panel(ax_path, ax_var, gp, mpc, tA, tB,
                    pA, pB, chose_A, it):
    ax_path.clear()
    draw_vessels(ax_path, mpc)

    cA   = "#534AB7" if chose_A else "#B4B2A9"
    cB   = "#B4B2A9" if chose_A else "#534AB7"
    lwA, lwB = (2.5, 1.2) if chose_A else (1.2, 2.5)
    lbA  = "Path A [chosen]" if chose_A else "Path A"
    lbB  = "Path B" if chose_A else "Path B [chosen]"

    ax_path.plot(pA[:, 0], pA[:, 1], color=cA, lw=lwA, label=lbA, zorder=3)
    ax_path.plot(pB[:, 0], pB[:, 1], color=cB, lw=lwB, label=lbB,
                 ls="--", zorder=3)
    ax_path.plot(*mpc.start, "s", color="#1D9E75", ms=8,  zorder=5)
    ax_path.plot(*mpc.goal,  "*", color="#1D9E75", ms=12, zorder=5)
    ax_path.set_xlim(-60, 1060); ax_path.set_ylim(-280, 280)
    ax_path.set_aspect("equal")
    ax_path.set_title(f"Iter {it+1}: Path Comparison", fontsize=9)
    ax_path.legend(fontsize=7, loc="upper left")
    ax_path.grid(True, alpha=0.3)

    ax_var.clear()
    vars_ = gp.param_variances()
    bars  = ax_var.bar(THETA_NAMES, vars_, color=COLORS, alpha=0.8, zorder=3)
    ax_var.axhline(ENT_THRESH, color="#E24B4A", ls="--", lw=1.2,
                   label=f"thresh={ENT_THRESH}")
    ax_var.set_ylim(0, 1.05)
    ax_var.set_title(f"Param Variance (mean={np.mean(vars_):.3f})", fontsize=9)
    ax_var.legend(fontsize=7)
    ax_var.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vars_):
        ax_var.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.02, f"{v:.2f}", ha="center", fontsize=7)


def plot_final(gp, mpc, theta_star, history, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("GP-BPL + MPC: Final Results", fontsize=12, fontweight="bold")

    # 1. 최종 경로
    ax    = axes[0, 0]
    p_s   = mpc.generate_path(theta_star)
    p_t   = mpc.generate_path(TRUE_THETA)
    draw_vessels(ax, mpc)
    ax.plot(p_s[:, 0], p_s[:, 1], "#534AB7", lw=2.5, label="Est theta*")
    ax.plot(p_t[:, 0], p_t[:, 1], "#1D9E75", lw=2.5, ls="--", label="True theta")
    ax.plot(*mpc.start, "s", color="#1D9E75", ms=8)
    ax.plot(*mpc.goal,  "*", color="#1D9E75", ms=12)
    ax.set_xlim(-60, 1060); ax.set_ylim(-280, 280)
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("Final Path (Est vs True)"); ax.grid(True, alpha=0.3)

    # 2. 파라미터 비교
    ax = axes[0, 1]
    x  = np.arange(len(THETA_NAMES))
    w  = 0.35
    ax.bar(x - w/2, TRUE_THETA, w, label="True",    color="#1D9E75", alpha=0.8)
    ax.bar(x + w/2, theta_star, w, label="Est theta*", color="#534AB7", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(THETA_NAMES)
    ax.legend(fontsize=9); ax.set_title("Parameter Estimation")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (t, s) in enumerate(zip(TRUE_THETA, theta_star)):
        err = abs(t - s) / abs(t) * 100
        ax.text(i, max(t, s) + 0.15, f"{err:.1f}%", ha="center", fontsize=8)

    # 3. 수렴 곡선 — iteration마다 감소해야 함
    ax    = axes[1, 0]
    iters = [h["it"] + 1 for h in history]
    mvars = [h["mean_var"] for h in history]
    ax.plot(iters, mvars, "o-", color="#534AB7", lw=2.5, ms=7,
            label="mean variance")
    ax.axhline(ENT_THRESH, color="#E24B4A", ls="--", lw=1.5,
               label=f"convergence ({ENT_THRESH})")
    ax.fill_between(iters, mvars, ENT_THRESH,
                    where=[v > ENT_THRESH for v in mvars],
                    alpha=0.15, color="#534AB7")
    # 감소 여부 확인
    monotone = all(mvars[i] >= mvars[i+1] for i in range(len(mvars)-1))
    ax.set_title(f"GP Convergence {'(monotone ✓)' if monotone else '(not monotone)'}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Posterior Variance")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. Confidence
    ax = axes[1, 1]
    for h in history:
        for i, name in enumerate(THETA_NAMES):
            conf = h["parsed"][name]["confidence"]
            ax.scatter(h["it"] + 1, conf, color=COLORS[i], s=60,
                       zorder=3, alpha=0.85)
    ax.axhline(0.6, color="gray", ls="--", lw=1, label="high-conf threshold")
    handles = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS, THETA_NAMES)]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Confidence")
    ax.set_title("Per-Query Confidence"); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("GP-BPL + MPC Maritime Style Learning")
    print(f"True theta : {dict(zip(THETA_NAMES, TRUE_THETA))}")
    print(f"Convergence: mean_var < {ENT_THRESH}")
    print("=" * 60)

    gp  = GPBPL(dim=4, bounds=THETA_BOUNDS, var_noise=1.0)
    mpc = MPCSim()
    os.makedirs("./outputs", exist_ok=True)

    n_cols   = 2
    n_rows   = (MAX_ITER + n_cols - 1) // n_cols
    fig_it, axes_it = plt.subplots(
        n_rows, n_cols * 2, figsize=(16, 4.2 * n_rows)
    )
    fig_it.suptitle("GP-BPL Iterations", fontsize=12, fontweight="bold")

    history   = []
    converged = False

    for it in range(MAX_ITER):
        print(f"\n── Iteration {it+1} ──")

        # 1. Acquisition
        tA, tB = gp.acquisition()
        print(f"  tA: {dict(zip(THETA_NAMES, tA.round(2)))}")
        print(f"  tB: {dict(zip(THETA_NAMES, tB.round(2)))}")

        # 2. MPC 경로 생성
        pA, pB = mpc.generate_path(tA), mpc.generate_path(tB)
        mA, mB = mpc.metrics(pA), mpc.metrics(pB)
        print(f"  PathA: clearance={mA['min_clearance_m']}m  "
              f"smooth={mA['smoothness']:.3f}  len={mA['path_length_m']}m")
        print(f"  PathB: clearance={mB['min_clearance_m']}m  "
              f"smooth={mB['smoothness']:.3f}  len={mB['path_length_m']}m")

        # 3. LLM 경로 설명 (더미)
        expl = llm_explain(tA, tB, mA, mB)
        print(f"  [LLM] {expl}")

        # 4. 더미 운항자 응답
        chose_A, resp = dummy_operator(tA, tB, it)
        print(f"  [Operator] {'chose A' if chose_A else 'chose B'}: {resp['text']}")

        # 5. LLM Reasoning 파싱 (더미)
        parsed = llm_parse(resp)
        print("  [Parse]")
        for name in THETA_NAMES:
            info = parsed[name]
            if info["confidence"] > 0:
                print(f"    {name:6s} dir={info['direction']:.2f}  "
                      f"conf={info['confidence']:.2f}  ({info['evidence']})")

        # 6. GP 업데이트
        gp.update_pairwise(tA, tB, chose_A)
        gp.update_reasoning(parsed)

        mean_var = gp.mean_variance()
        pv       = gp.param_variances()
        print(f"  Mean Var: {mean_var:.4f}  "
              f"| per-param: {[f'{v:.3f}' for v in pv]}")

        history.append({
            "it": it, "tA": tA, "tB": tB,
            "chose_A": chose_A, "parsed": parsed, "mean_var": mean_var,
        })

        row = it // n_cols
        col = (it % n_cols) * 2
        if row < axes_it.shape[0] and col + 1 < axes_it.shape[1]:
            plot_iter_panel(
                axes_it[row, col], axes_it[row, col + 1],
                gp, mpc, tA, tB, pA, pB, chose_A, it
            )

        # 7. 수렴 판단
        if mean_var < ENT_THRESH:
            print(f"\n  ✓ Converged at iteration {it+1}")
            converged = True
            break

    # 빈 서브플롯 숨기기
    for r in range(n_rows):
        for c in range(n_cols * 2):
            try:
                if not axes_it[r, c].lines and not axes_it[r, c].patches:
                    axes_it[r, c].set_visible(False)
            except IndexError:
                pass

    iter_path = "./outputs/gp_bpl_iterations.png"
    fig_it.tight_layout()
    fig_it.savefig(iter_path, dpi=130, bbox_inches="tight")
    plt.close(fig_it)

    # θ* 확정
    theta_star = gp.theta_star()
    print("\n" + "=" * 60)
    print("Final theta* Estimation")
    print("=" * 60)
    total_err = 0.0
    for name, true, est in zip(THETA_NAMES, TRUE_THETA, theta_star):
        err        = abs(true - est) / abs(true) * 100
        total_err += err
        print(f"  {name:6s}  True={true:.2f}  Est={est:.2f}  err={err:.1f}%")
    print(f"  Mean error: {total_err / len(THETA_NAMES):.1f}%")

    if not converged:
        print(f"  (Not converged within {MAX_ITER} iterations)")

    result_path = "./outputs/gp_bpl_result.png"
    plot_final(gp, mpc, theta_star, history, result_path)

    print(f"\nSaved:")
    print(f"  Iterations : {iter_path}")
    print(f"  Result     : {result_path}")


if __name__ == "__main__":
    main()