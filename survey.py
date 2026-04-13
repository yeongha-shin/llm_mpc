import numpy as np
import matplotlib.pyplot as plt
from modules.preference.gp_bpl import GPBPL


def visualize(gp, iteration):
    theta_range = gp.get_theta_range()
    mu, var = gp.posterior(theta_range)

    plt.clf()
    plt.plot(theta_range, mu, 'b-', label='posterior mean', linewidth=2)
    plt.fill_between(theta_range,
                     mu - 2 * np.sqrt(var),
                     mu + 2 * np.sqrt(var),
                     alpha=0.3, label='uncertainty (2σ)')

    # 현재 best 표시
    best = gp.get_current_best()
    plt.axvline(x=best, color='r', linestyle='--', label=f'current best: {best:.4f}')

    # 관측 포인트 표시
    for theta_A, theta_B, pref in gp.observations:
        if pref == 1:
            plt.axvline(x=theta_A, color='g', alpha=0.3, linewidth=1)
        else:
            plt.axvline(x=theta_B, color='g', alpha=0.3, linewidth=1)

    plt.xlabel('tc_cbf_gain')
    plt.ylabel('preference')
    plt.title(f'Iteration {iteration} | observations: {len(gp.observations)}')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)


def is_converged(gp, threshold=0.5):
    """
    종료 조건:
    uncertainty가 충분히 낮으면 종료
    """
    theta_range = gp.get_theta_range()
    _, var = gp.posterior(theta_range)
    max_var = np.max(var)
    return max_var < threshold


def run_survey():
    gp = GPBPL()

    plt.ion()
    plt.figure(figsize=(10, 5))

    max_iterations = 20

    print("=" * 50)
    print("선박 운항 스타일 설문 시작")
    print("tc_cbf_gain: 낮을수록 보수적(일찍 회피), 높을수록 공격적(늦게 회피)")
    print("=" * 50)

    for i in range(1, max_iterations + 1):
        print(f"\n[질문 {i}/{max_iterations}]")

        # 다음 질문 선택
        theta_A, theta_B = gp.next_query()

        # 사람에게 질문
        print(f"\n방식 A: tc_cbf_gain = {theta_A:.4f} (장애물 회피 시작 타이밍)")
        print(f"방식 B: tc_cbf_gain = {theta_B:.4f} (장애물 회피 시작 타이밍)")
        print(f"\n  A: {theta_A:.4f} → {'보수적 (일찍 회피)' if theta_A < 0.025 else '공격적 (늦게 회피)'}")
        print(f"  B: {theta_B:.4f} → {'보수적 (일찍 회피)' if theta_B < 0.025 else '공격적 (늦게 회피)'}")

        while True:
            answer = input("\n어느 방식을 선호하시나요? (A/B): ").strip().upper()
            if answer in ['A', 'B']:
                break
            print("A 또는 B를 입력해주세요.")

        preference = 1 if answer == 'A' else 0
        gp.add_observation(theta_A, theta_B, preference)

        # 현재 best 출력
        best = gp.get_current_best()
        print(f"\n현재 추정 best tc_cbf_gain: {best:.4f}")

        # 시각화
        visualize(gp, i)

        # 종료 조건 확인
        if is_converged(gp, threshold=0.5):
            print(f"\n수렴 완료! {i}번의 질문으로 선호 모델 학습 완료")
            break

    plt.ioff()

    # 최종 결과
    final_best = gp.get_current_best()
    print("\n" + "=" * 50)
    print("설문 완료!")
    print(f"학습된 tc_cbf_gain: {final_best:.4f}")
    print(f"총 질문 수: {len(gp.observations)}")
    print("=" * 50)

    plt.show()
    return final_best


if __name__ == '__main__':
    theta_star = run_survey()
    print(f"\ntheta* = {theta_star:.4f}")