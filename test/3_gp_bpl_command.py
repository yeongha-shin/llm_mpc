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


# ──────────────────────────────────────────
# Step 2: 운항 명령 처리
# ──────────────────────────────────────────

# 테스트용 자연어 명령 시나리오
COMMAND_SCENARIOS = [
    {
        "text": "A 지점으로 가줘",
        "intent": {"q_pos": None, "q_vel": None, "r_n": None, "S": None},
        "goal": np.array([1000.0, 0.0]),
        "expected": "normal",   # θ* 그대로 사용
    },
    {
        "text": "최대한 빨리 A 지점으로 가줘",
        "intent": {"q_pos": None, "q_vel": 2.8, "r_n": None, "S": None},
        "goal": np.array([1000.0, 0.0]),
        "expected": "conflict",  # q_vel이 θ*보다 높음 → 충돌
    },
    {
        "text": "COLREG 무시하고 최단거리로 가줘",
        "intent": {"q_pos": None, "q_vel": 2.5, "r_n": None, "S": 1.0},
        "goal": np.array([1000.0, 0.0]),
        "expected": "conflict",  # S가 θ*보다 훨씬 낮음 → 충돌
    },
    {
        "text": "안전하게 천천히 가줘",
        "intent": {"q_pos": 4.5, "q_vel": 0.5, "r_n": None, "S": None},
        "goal": np.array([1000.0, 0.0]),
        "expected": "normal",   # θ*와 같은 방향
    },
    {
        "text": "긴급 상황, 즉시 최단거리로",
        "intent": {"q_pos": 0.5, "q_vel": 3.0, "r_n": None, "S": 0.5},
        "goal": np.array([1000.0, 0.0]),
        "expected": "emergency",  # θ* bypass
    },
]


def llm_parse_command(command_text, scenario):
    """
    자연어 명령 파싱 (더미)
    실제 LLM은 명령에서 의도된 파라미터 값을 추출
    반환: intent dict — None이면 θ* 그대로 사용
    """
    return scenario["intent"], scenario["goal"]


def detect_conflict(intent, theta_star, bounds, conflict_thresh=0.35):
    """
    명령 의도와 θ* 간 충돌 감지

    충돌 기준:
      명령에서 지정한 파라미터 값이 θ*와
      normalized 거리로 conflict_thresh 이상 차이날 때

    반환:
      conflicts: [(param_name, theta*_val, intent_val, gap), ...]
      severity: "none" | "minor" | "major" | "emergency"
    """
    conflicts = []
    lo, hi = bounds[:, 0], bounds[:, 1]

    for i, name in enumerate(THETA_NAMES):
        if intent[name] is None:
            continue  # 명령에서 언급 안 한 파라미터 → 충돌 없음

        norm_star   = (theta_star[i] - lo[i]) / (hi[i] - lo[i] + 1e-8)
        norm_intent = (intent[name]  - lo[i]) / (hi[i] - lo[i] + 1e-8)
        gap = abs(norm_star - norm_intent)

        if gap > conflict_thresh:
            conflicts.append({
                "param":    name,
                "star_val": theta_star[i],
                "cmd_val":  intent[name],
                "gap":      gap,
            })

    # 심각도 판단
    if not conflicts:
        severity = "none"
    elif max(c["gap"] for c in conflicts) > 0.7:
        severity = "emergency"
    elif len(conflicts) >= 2:
        severity = "major"
    else:
        severity = "minor"

    return conflicts, severity


def resolve_conflict(conflicts, severity, theta_star, intent, scenario):
    """
    충돌 처리 정책

    none      → θ* 그대로 + 명령 목표만 추가
    minor     → θ* 일부 조율 (intent 방향으로 살짝 이동)
    major     → 되질문 생성
    emergency → θ* bypass, 명령 우선
    """
    result_theta = theta_star.copy()

    if severity == "none":
        # 명령에서 지정한 값 반영 (θ*와 같은 방향)
        for i, name in enumerate(THETA_NAMES):
            if intent[name] is not None:
                result_theta[i] = intent[name]
        action = "direct"
        message = "명령이 선호 모델과 일치합니다. θ*로 운항합니다."

    elif severity == "minor":
        # θ*와 명령 사이 중간값 (0.7 θ* + 0.3 intent)
        for i, name in enumerate(THETA_NAMES):
            if intent[name] is not None:
                result_theta[i] = 0.7 * theta_star[i] + 0.3 * intent[name]
        action = "blend"
        param_str = ", ".join(c["param"] for c in conflicts)
        message = (f"[{param_str}] 파라미터가 선호 모델과 다소 다릅니다. "
                   f"자동으로 조율하여 운항합니다.")

    elif severity == "major":
        # 되질문 생성 — 운항자 확인 필요
        result_theta = theta_star.copy()  # 일단 θ* 유지
        action = "clarify"
        lines = []
        for c in conflicts:
            lines.append(
                f"  {c['param']}: 선호={c['star_val']:.2f}, "
                f"명령={c['cmd_val']:.2f} (차이 {c['gap']*100:.0f}%)"
            )
        message = ("선호 모델과 충돌하는 명령입니다. 확인이 필요합니다:\n"
                   + "\n".join(lines)
                   + "\n계속 진행하시겠습니까? (Y/N)")

    else:  # emergency
        # θ* bypass — 명령 최우선
        for i, name in enumerate(THETA_NAMES):
            if intent[name] is not None:
                result_theta[i] = intent[name]
        action = "bypass"
        message = ("⚠ 긴급 명령 감지. 선호 모델을 일시 중단하고 "
                   "명령을 최우선으로 실행합니다.")

    return result_theta, action, message


def run_command_phase(gp, mpc):
    """Step 2: 명령 처리 루프"""
    theta_star = gp.theta_star()

    print("\n" + "=" * 60)
    print("Step 2: 운항 명령 처리")
    print(f"확정된 θ* = {dict(zip(THETA_NAMES, theta_star.round(2)))}")
    print("=" * 60)

    results = []

    for i, scenario in enumerate(COMMAND_SCENARIOS):
        print(f"\n── Command {i+1}: \"{scenario['text']}\" ──")

        # 1. 명령 파싱
        intent, goal = llm_parse_command(scenario["text"], scenario)
        intent_str = {k: v for k, v in intent.items() if v is not None}
        print(f"  파싱된 의도: {intent_str}")

        # 2. 충돌 감지
        conflicts, severity = detect_conflict(
            intent, theta_star, THETA_BOUNDS
        )
        print(f"  충돌 심각도: {severity.upper()}")
        if conflicts:
            for c in conflicts:
                print(f"    {c['param']}: θ*={c['star_val']:.2f}  "
                      f"cmd={c['cmd_val']:.2f}  gap={c['gap']*100:.0f}%")

        # 3. 충돌 처리
        result_theta, action, message = resolve_conflict(
            conflicts, severity, theta_star, intent, scenario
        )
        print(f"  처리 방식: {action.upper()}")
        print(f"  메시지: {message}")
        print(f"  실행 θ: {dict(zip(THETA_NAMES, result_theta.round(2)))}")

        # 4. MPC 경로 생성
        path = mpc.generate_path(result_theta)
        m    = mpc.metrics(path)
        print(f"  경로: clearance={m['min_clearance_m']}m  "
              f"smooth={m['smoothness']:.3f}  len={m['path_length_m']}m")

        results.append({
            "scenario":   scenario,
            "intent":     intent,
            "conflicts":  conflicts,
            "severity":   severity,
            "action":     action,
            "result_theta": result_theta,
            "path":       path,
            "metrics":    m,
        })

    return theta_star, results


def plot_command_results(theta_star, results, mpc, out_path):
    """명령 처리 결과 시각화"""
    n     = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 9))
    fig.suptitle("Step 2: Command Processing Results", fontsize=12,
                 fontweight="bold")

    # 색상: severity별
    sev_colors = {
        "none":      "#1D9E75",
        "minor":     "#EF9F27",
        "major":     "#E24B4A",
        "emergency": "#A32D2D",
    }

    # θ* 기준 경로
    path_star = mpc.generate_path(theta_star)

    for i, r in enumerate(results):
        ax_path = axes[0, i]
        ax_bar  = axes[1, i]

        sev   = r["severity"]
        color = sev_colors[sev]

        # 경로 그림
        draw_vessels(ax_path, mpc)
        ax_path.plot(path_star[:, 0], path_star[:, 1],
                     color="#B4B2A9", lw=1.5, ls="--", label="θ* base", zorder=2)
        ax_path.plot(r["path"][:, 0], r["path"][:, 1],
                     color=color, lw=2.5, label=f"{r['action']}", zorder=3)
        ax_path.plot(*mpc.start, "s", color="#1D9E75", ms=7, zorder=5)
        ax_path.plot(*mpc.goal,  "*", color="#1D9E75", ms=11, zorder=5)
        ax_path.set_xlim(-60, 1060); ax_path.set_ylim(-280, 280)
        ax_path.set_aspect("equal")
        ax_path.set_title(
            f"Cmd {i+1}: {r['scenario']['text'][:22]}\n"
            f"[{sev.upper()}] → {r['action']}",
            fontsize=8, color=color, fontweight="bold"
        )
        ax_path.legend(fontsize=7, loc="upper left")
        ax_path.grid(True, alpha=0.3)

        # 파라미터 비교 바
        x = np.arange(len(THETA_NAMES))
        w = 0.28
        ax_bar.bar(x - w, theta_star,        w, label="θ*",
                   color="#534AB7", alpha=0.8)
        ax_bar.bar(x,     r["result_theta"], w, label="executed",
                   color=color, alpha=0.8)
        intent_vals = [r["intent"][n] if r["intent"][n] is not None
                       else np.nan for n in THETA_NAMES]
        ax_bar.bar(x + w, intent_vals, w, label="cmd intent",
                   color="#888780", alpha=0.6)

        ax_bar.set_xticks(x); ax_bar.set_xticklabels(THETA_NAMES, fontsize=8)
        ax_bar.legend(fontsize=7); ax_bar.grid(True, alpha=0.3, axis="y")
        ax_bar.set_title(f"Parameter Comparison", fontsize=8)

        # 충돌 파라미터 표시
        for c in r["conflicts"]:
            idx = THETA_NAMES.index(c["param"])
            ax_bar.axvline(idx, color="#E24B4A", alpha=0.3, lw=8, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n명령 처리 결과 저장: {out_path}")




# ──────────────────────────────────────────
# main() 에 Step 2 연결
# ──────────────────────────────────────────
def main_full():
    """Step 1 (GP-BPL 설문) + Step 2 (명령 처리) 통합 실행"""

    # ── Step 1 ──────────────────────────────
    print("=" * 60)
    print("Step 1: GP-BPL 운항 스타일 학습")
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
        tA, tB  = gp.acquisition()
        pA, pB  = mpc.generate_path(tA), mpc.generate_path(tB)
        mA, mB  = mpc.metrics(pA), mpc.metrics(pB)
        chose_A, resp = dummy_operator(tA, tB, it)
        parsed  = llm_parse(resp)

        gp.update_pairwise(tA, tB, chose_A)
        gp.update_reasoning(parsed)

        mean_var = gp.mean_variance()
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

        print(f"  Iter {it+1}: mean_var={mean_var:.4f}")

        if mean_var < ENT_THRESH:
            print(f"  ✓ Converged at iteration {it+1}")
            converged = True
            break

    for r in range(n_rows):
        for c in range(n_cols * 2):
            try:
                if not axes_it[r, c].lines and not axes_it[r, c].patches:
                    axes_it[r, c].set_visible(False)
            except IndexError:
                pass

    fig_it.tight_layout()
    fig_it.savefig("./outputs/gp_bpl_iterations.png", dpi=130,
                   bbox_inches="tight")
    plt.close(fig_it)

    theta_star = gp.theta_star()
    plot_final(gp, mpc, theta_star, history, "./outputs/gp_bpl_result.png")

    print(f"\n확정 θ*: {dict(zip(THETA_NAMES, theta_star.round(2)))}")

    # ── Step 2 ──────────────────────────────
    theta_star, cmd_results = run_command_phase(gp, mpc)
    plot_command_results(
        theta_star, cmd_results, mpc,
        "./outputs/gp_bpl_commands.png"
    )

    print("\n" + "=" * 60)
    print("완료. 저장 파일:")
    print("  ./outputs/gp_bpl_iterations.png")
    print("  ./outputs/gp_bpl_result.png")
    print("  ./outputs/gp_bpl_commands.png")


if __name__ == "__main__":
    main_full()