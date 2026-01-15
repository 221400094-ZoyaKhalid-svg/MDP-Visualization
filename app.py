import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MDP Visualization",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
body {
    background-color: #f5f7fb;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.15);
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.9;
}

/* Section titles */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e293b;
    margin: 2rem 0 1rem;
}

/* Buttons */
button {
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
}

/* Slider */
.css-1y4p8pa {
    padding-top: 10px;
}

/* Legend */
.legend {
    background-color: #ffffff;
    padding: 0.7rem 1.2rem;
    border-radius: 12px;
    font-size: 0.9rem;
    color: #334155;
    margin-top: 1.2rem;
    text-align: center;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

ROWS, COLS = 5, 5
GOAL_STATE = (4, 4)
NEGATIVE_STATE = (3, 3)
OBSTACLES = [(1, 1), (2, 2)]

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

ACTION_SYMBOLS = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

STEP_REWARD = -0.1
GOAL_REWARD = 10
NEGATIVE_REWARD = -10


def is_valid_state(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and state not in OBSTACLES

def get_next_state(state, action):
    dr, dc = ACTIONS[action]
    next_state = (state[0] + dr, state[1] + dc)
    return next_state if is_valid_state(next_state) else state

def get_reward(state):
    if state == GOAL_STATE:
        return GOAL_REWARD
    if state == NEGATIVE_STATE:
        return NEGATIVE_REWARD
    return STEP_REWARD

def value_iteration(gamma, iterations):
    V = np.zeros((ROWS, COLS))
    history = []

    for _ in range(iterations):
        new_V = np.copy(V)
        for r in range(ROWS):
            for c in range(COLS):
                s = (r, c)
                if s in OBSTACLES or s in [GOAL_STATE, NEGATIVE_STATE]:
                    continue
                new_V[s] = max(
                    get_reward(get_next_state(s, a)) + gamma * V[get_next_state(s, a)]
                    for a in ACTIONS
                )
        V = new_V
        history.append(np.copy(V))
    return history

def policy_iteration(gamma, max_iterations):
    policy = {
        (r, c): np.random.choice(list(ACTIONS.keys()))
        for r in range(ROWS) for c in range(COLS)
        if (r, c) not in OBSTACLES and (r, c) not in [GOAL_STATE, NEGATIVE_STATE]
    }

    V = np.zeros((ROWS, COLS))
    history = []

    for _ in range(max_iterations):
        for _ in range(20):
            for s, a in policy.items():
                ns = get_next_state(s, a)
                V[s] = get_reward(ns) + gamma * V[ns]

        history.append(np.copy(V))

        stable = True
        for s in policy:
            old = policy[s]
            policy[s] = max(
                ACTIONS,
                key=lambda a: get_reward(get_next_state(s, a)) + gamma * V[get_next_state(s, a)]
            )
            if old != policy[s]:
                stable = False

        if stable:
            break

    return history, policy

def plot_grid(V, policy=None):
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.imshow(V, cmap="coolwarm")

    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c)
            if s in OBSTACLES:
                ax.text(c, r, "X", ha="center", va="center", fontsize=18)
            elif s == GOAL_STATE:
                ax.text(c, r, "G", ha="center", va="center", fontsize=18, color="green")
            elif s == NEGATIVE_STATE:
                ax.text(c, r, "-10", ha="center", va="center", fontsize=14, color="red")
            else:
                ax.text(c, r, f"{V[s]:.1f}", ha="center", va="center")
                if policy and s in policy:
                    ax.text(c, r + 0.35, ACTION_SYMBOLS[policy[s]], ha="center", fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)


st.session_state.setdefault("history", None)
st.session_state.setdefault("policy", None)
st.session_state.setdefault("step", 1)


st.markdown("""
<div class="hero">
    <h1>Markov Decision Process</h1>
    <p>Interactive Visualization of Value Iteration & Policy Iteration</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>⚙ Algorithm Settings</div>", unsafe_allow_html=True)

algorithm = st.selectbox("Algorithm", ["Value Iteration", "Policy Iteration"])
gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)
iterations = st.slider("Maximum Iterations", 1, 50, 10)

c1, c2 = st.columns(2)
with c1:
    if st.button("▶ Run Algorithm", use_container_width=True):
        st.session_state.step = 1
        if algorithm == "Value Iteration":
            st.session_state.history = value_iteration(gamma, iterations)
            st.session_state.policy = None
        else:
            st.session_state.history, st.session_state.policy = policy_iteration(gamma, iterations)

with c2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.history = None
        st.session_state.policy = None
        st.session_state.step = 1


if st.session_state.history:
    st.markdown("<div class='section-title'>📊 Iteration Progress</div>", unsafe_allow_html=True)

    max_step = len(st.session_state.history)
    a, b, c = st.columns([1, 2, 1])

    with a:
        if st.button("⬅ Previous"):
            st.session_state.step = max(1, st.session_state.step - 1)
    with c:
        if st.button("Next ➡"):
            st.session_state.step = min(max_step, st.session_state.step + 1)
    with b:
        st.session_state.step = st.slider(
            "Iteration Step",
            1,
            max_step,
            st.session_state.step
        )

    st.markdown(f"### Iteration {st.session_state.step}")
    plot_grid(st.session_state.history[st.session_state.step - 1], st.session_state.policy)

    st.markdown("""
    <div class="legend">
    🟩 Goal State &nbsp;&nbsp; 🟥 Negative State &nbsp;&nbsp; ⬛ Obstacle
    </div>
    """, unsafe_allow_html=True)
