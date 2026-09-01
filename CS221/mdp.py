from dataclasses import dataclass
import numpy as np
from typing import Any, List, Callable
from collections import defaultdict
from graphviz import Digraph

@dataclass(frozen=True)
class Step:
    action: Any
    reward: float # Reward R(s, a, s')
    prob: float # State transition probability T(s, a, s')
    state: Any # Next state s'

class MDP:
    def start_state(self) -> Any:
        raise NotImplementedError
    
    def successors(self, state: Any) -> List[Step]:
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        raise NotImplementedError

    def discount(self) -> float:
        raise NotImplementedError

# Policy: map state (input) to action (output)
Policy = Callable[[Any], Any]

@dataclass
class Rollout:
    steps: List[Step]
    discount: float
    utility: float

    def __init__(self, steps: List[Step], discount: float):
        self.steps = steps
        self.discount = discount
        self.utility = sum(step.reward * discount ** i for i, step in enumerate(steps))

class FlackyTramMDP(MDP):
    def __init__(self, num_locs: int = 20, failure_prob: float = 0.4):
        self.num_locs = num_locs
        self.failure_prob = failure_prob

    def start_state(self) -> int:
        return 1

    def successors(self, state: int) -> List[Step]:
        steps = []
        # Action 1: walk (takes 1 min, reward -1, 100% success)
        if state + 1 <= self.num_locs:
            steps.append(Step(action="walk", reward=-1.0, prob=1.0, state=state+1))
        # Action 2: tram (takes 2 min, reward -2, 60% success)
        if state * 2 <= self.num_locs:
            steps.append(Step(action="tram", reward=-2, prob=1-self.failure_prob, state=state * 2))
            steps.append(Step(action="tram", reward=-2, prob=self.failure_prob, state=state))
        else:
            #Invalid action, add penalty -100, end at last stop
            steps.append(Step(action="tram", reward=-100, prob=1, state=self.num_locs))
        return steps

    def is_end(self, state: int) -> bool:
        return state == self.num_locs

    def discount(self) -> float:
        return 0.9

@dataclass(frozen=True)
class ValueIterationResult:
    values: dict[Any, float] # map from state to Q value
    pi: dict[Any, Any] # optimal policy. map from state to action
    distances: List[float] # distances from each iteration

def get_states(mdp: MDP) -> List[Any]:
    states = set()
    def next_state(state):
        if state in states:
            return
        states.add(state)
        for step in mdp.successors(state):
            next_state(step.state)

    next_state(mdp.start_state())
    return list(states)

def initial_values(mdp: MDP, states: List[Any]) -> dict:
    return {s: 0 if mdp.is_end(s) else -100 for s in states}

def compute_q_value(values: dict[Any, float], steps: List[Step], discount: float) -> float:
    # compute Q(s, a) = ∑ T(s,a,s') * [R(s,a,s') + γ * V(s')]
    return sum(step.prob * (step.reward + discount * values[step.state]) for step in steps)

def compute_distance(values: dict, next_values: dict) -> float:
    return max([abs(values[s] - next_values[s]) for s in values.keys()])

def get_action_successors(mdp: MDP, state: Any) -> dict[Any, List[Step]]:
    action_to_successors = defaultdict(list)
    for step in mdp.successors(state):
        action_to_successors[step.action].append(step)
    return action_to_successors

def value_iteration_for_state(mdp: MDP, state: Any, values: dict[Any, float]) -> tuple[float, Any]:
    actions = []
    utilities = []
    for action, steps in get_action_successors(mdp, state).items():
        actions.append(action)
        utilities.append(compute_q_value(values, steps, mdp.discount()))
    return max(utilities), actions[np.argmax(utilities)]

# Take MDP and return optimal policy with value iteration
def value_evaluation(mdp: MDP, max_iters: int = 100, tolerance: float = 1e-5) -> ValueIterationResult:
    states = get_states(mdp)
    values = initial_values(mdp, states)
    distances = []
    pi = {s: None for s in states}
    for _ in range(max_iters):
        next_values = {}
        for s in states:
            if mdp.is_end(s):
                next_values[s] = 0
                continue
            val, action = value_iteration_for_state(mdp, s, values)
            next_values[s] = val
            pi[s] = action
        distance = compute_distance(values, next_values)
        distances.append(distance)
        values = next_values
        if distance < tolerance:
            break
    return ValueIterationResult(values, pi, distances)

def draw_mdp_graph(mdp: MDP):
    dot = Digraph(comment="MDP Graph")
    visited = set()
    def recurse(state):
        if state in visited:
            return
        visited.add(state)

        if mdp.is_end(state):
            dot.node(str(state), shape="doublecircle", color="green")
        else:
            dot.node(str(state), shape='circle')
            action_to_steps = defaultdict(list)
            for successor in mdp.successors(state):
                action_to_steps[successor.action].append(successor)
            for action, steps in action_to_steps.items():
                act_node_id = f"{state}_{action}"
                dot.node(act_node_id,label=action, shape="point")
                dot.edge(str(state), act_node_id, style="dashed")

                for step in steps:
                    label = f"p={step.prob:.2f}\nr={step.reward}"
                    dot.edge(act_node_id, str(step.state), label=label)
                    recurse(step.state)
    recurse(mdp.start_state())
    return dot