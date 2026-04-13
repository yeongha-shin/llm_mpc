import numpy as np
import matplotlib.pyplot as plt
from modules.preference.gp_bpl import GPBPL
import pickle

def visualize(gp, current_context, iteration, fig):
    contexts = [
        [500, 5, "가깝고 복잡 (dist=500m, n_obs=5)"],
        [500, 1, "가깝고 단순 (dist=500m, n_obs=1)"],
        [2000, 5, "멀고 복잡 (dist=2000m, n_obs=5)"],
        [2000, 1, "멀고 단순 (dist=2000m, n_obs=1)"],
    ]

    fig.clf()  # 기존 figure 초기화
    axes = fig.subplots(2, 2)
    fig.suptitle(f'Iteration {iteration} | 현재 상황: dist={current_context[0]}m, n_obs={current_context[1]}')

    theta_range = gp.get_theta_range()

    for ax, (dist, n_obs, title) in zip(axes.flatten(), contexts):
        ctx = [dist, n_obs]
        X_test = np.array([
            gp._normalize(t, ctx[0], ctx[1])
            for t in theta_range
        ])
        mu, var = gp.posterior(X_test)
        best = gp.get_current_best(ctx)

        ax.plot(theta_range, mu, 'b-', label='posterior mean', linewidth=2)
        ax.fill_between(theta_range,
                        mu - 2 * np.sqrt(var),
                        mu + 2 * np.sqrt(var),
                        alpha=0.3, label='uncertainty (2σ)')
        ax.axvline(x=best, color='r', linestyle='--',
                   label=f'best: {best:.4f}')

        if dist == current_context[0] and n_obs == current_context[1]:
            ax.set_facecolor('#fffbe6')

        ax.set_xlabel('tc_cbf_gain')
        ax.set_ylabel('preference')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.pause(0.1)

def is_converged(gp, context, threshold=0.3):
    theta_range = gp.get_theta_range()
    X_test = np.array([
        gp._normalize(t, context[0], context[1])
        for t in theta_range
    ])
    _, var = gp.posterior(X_test)
    return np.max(var) < threshold


def run_survey():
    gp = GPBPL()

    plt.ion()
    fig = plt.figure(figsize=(14, 8))  # 한번만 생성

    max_iterations = 5

    print("=" * 50)
    print("선박 운항 스타일 설문 시작")
    print("=" * 50)

    for i in range(1, max_iterations + 1):
        print(f"\n[질문 {i}/{max_iterations}]")

        # 시나리오 context 랜덤 생성

        # 다음 질문 선택
        (theta_A, theta_B), context = gp.next_query()
        min_dist = context[0]
        n_obstacles = context[1]

        print(f"\n상황: 장애물 {n_obstacles}척, 최소 거리 {min_dist}m")


        print(f"\n  A: tc_cbf_gain = {theta_A:.4f} → "
              f"{'보수적 (일찍 회피)' if theta_A < 0.025 else '공격적 (늦게 회피)'}")
        print(f"  B: tc_cbf_gain = {theta_B:.4f} → "
              f"{'보수적 (일찍 회피)' if theta_B < 0.025 else '공격적 (늦게 회피)'}")

        while True:
            answer = input("\n어느 방식을 선호하시나요? (A/B): ").strip().upper()
            if answer in ['A', 'B']:
                break
            print("A 또는 B를 입력해주세요.")

        preference = 1 if answer == 'A' else 0
        gp.add_observation(theta_A, context, theta_B, context, preference)

        best = gp.get_current_best(context)
        print(f"\n현재 상황에서 추정 best tc_cbf_gain: {best:.4f}")

        visualize(gp, context, i, fig)

        if is_converged(gp, context, threshold=0.3):
            print(f"\n수렴 완료! {i}번의 질문으로 학습 완료")
            break

    plt.ioff()

    save_path = 'gp_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(gp, f)
    print(f"\nGP 모델 저장 완료: {save_path}")

    return gp

    # # 최종 결과 출력 (다양한 context에서)
    # print("\n" + "=" * 50)
    # print("설문 완료! 상황별 학습된 tc_cbf_gain:")
    # test_contexts = [
    #     [500, 5],  # 가깝고 복잡
    #     [500, 1],  # 가깝고 단순
    #     [2000, 5],  # 멀고 복잡
    #     [2000, 1],  # 멀고 단순
    #     [1250, 3],  # 중간
    # ]
    # for ctx in test_contexts:
    #     best = gp.get_current_best(ctx)
    #     print(f"  dist={ctx[0]}m, n_obs={ctx[1]}척 → tc_cbf_gain={best:.4f}")
    # print("=" * 50)
    #
    # plt.show()


if __name__ == '__main__':
    run_survey()