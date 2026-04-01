import numpy as np
import random

np.random.seed(0)


# ------------------------------------------------------------
# 1. Scenario generator
# ------------------------------------------------------------

def generate_scenario():

    scenario = {
        "CPA": random.uniform(0.3, 1.5),
        "TCPA": random.uniform(2, 10),
        "encounter": random.choice(["crossing", "head_on"])
    }

    actions = [
        {"type": "turn_starboard", "timing": "early"},
        {"type": "maintain_course"},
        {"type": "reduce_speed"}
    ]

    return scenario, actions


# ------------------------------------------------------------
# 2. Feature extractor
# ------------------------------------------------------------

def extract_features(scenario, action):

    safety = scenario["CPA"]

    if action["type"] == "turn_starboard":
        timing = 1 if action.get("timing") == "early" else 0
    else:
        timing = 0

    speed = 1 if action["type"] == "reduce_speed" else 0

    return np.array([safety, timing, speed])


# ------------------------------------------------------------
# 3. Synthetic navigator (ground truth style)
# ------------------------------------------------------------

class SyntheticNavigator:

    def __init__(self, theta):

        self.theta = theta

    def choose(self, phi_a, phi_b):

        u_a = self.theta @ phi_a
        u_b = self.theta @ phi_b

        p = np.exp(u_a) / (np.exp(u_a) + np.exp(u_b))

        if np.random.rand() < p:
            return 0   # choose A
        else:
            return 1   # choose B


# ------------------------------------------------------------
# 4. Bayesian Preference Learning (grid approximation)
# ------------------------------------------------------------

class BayesianPreferenceLearning:

    def __init__(self, grid):

        self.grid = grid
        self.posterior = np.ones(len(grid)) / len(grid)

    def update(self, phi_a, phi_b, choice):

        likelihood = []

        for theta in self.grid:

            u_a = theta @ phi_a
            u_b = theta @ phi_b

            p = 1 / (1 + np.exp(-(u_a - u_b)))

            if choice == 0:
                likelihood.append(p)
            else:
                likelihood.append(1 - p)

        likelihood = np.array(likelihood)

        self.posterior *= likelihood
        self.posterior /= np.sum(self.posterior)

    def estimate(self):

        return np.average(self.grid, axis=0, weights=self.posterior)


# ------------------------------------------------------------
# 5. Query selection (random for now)
# ------------------------------------------------------------

def select_query(actions):

    a, b = random.sample(actions, 2)

    return a, b


# ------------------------------------------------------------
# 6. Main simulation
# ------------------------------------------------------------

def run_simulation():

    # 가상의 숨겨진 항해사의 스타일을, 추정해 나가는게 중요함
    # TODO: colreg vs mission, efficiency

    true_theta = np.array([0.8, 0.6, -0.4])
    # safety, timing, speed reduction 선호도

    navigator = SyntheticNavigator(true_theta)

    grid = np.random.randn(500,3)

    bpl = BayesianPreferenceLearning(grid)

    print("True style:", true_theta)
    print()

    for step in range(30):
        # 시나리오 생성
        scenario, actions = generate_scenario()
        print("scenario:", scenario, "actions:", actions)

        # 2. 액션 선택
        a, b = select_query(actions)

        phi_a = extract_features(scenario, a)
        phi_b = extract_features(scenario, b)

        # 3. 항해사가 선택함
        choice = navigator.choose(phi_a, phi_b)

        # 4. BPL 업데이트함
        bpl.update(phi_a, phi_b, choice)

        # 5. 스타일 추산함
        estimate = bpl.estimate()

        print("Step", step+1)
        print("Estimate:", np.round(estimate,3))
        print()


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    run_simulation()