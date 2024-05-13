import numpy as np
import random
from pyMaze import maze, COLOR, agent

class QLearningAgent:
    def __init__(self, maze, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.maze = maze
        self.epsilon = epsilon  # Probabilidad de exploración
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.Q = {}  # Tabla Q

        # Inicializar la tabla Q con valores arbitrarios
        for row in range(maze.size[0]):
            for col in range(maze.size[1]):
                for action in range(4):  # 4 acciones posibles: arriba, abajo, izquierda, derecha
                    self.Q[((row, col), action)] = 0.0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Exploración aleatoria
        else:
            # Exploit: elegir la mejor acción según la tabla Q
            max_actions = []
            max_q_value = float("-inf")
            for action in range(4):
                if self.Q[(state, action)] > max_q_value:
                    max_actions = [action]
                    max_q_value = self.Q[(state, action)]
                elif self.Q[(state, action)] == max_q_value:
                    max_actions.append(action)
            return random.choice(max_actions)

    def learn(self, state, action, reward, next_state):
        old_q_value = self.Q[(state, action)]
        max_next_q_value = max([self.Q[(next_state, a)] for a in range(4)])
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_next_q_value)
        self.Q[(state, action)] = new_q_value

def run_with_q_learning():
    m = maze(10, 30)
    m.CreateMaze()

    q_agent = QLearningAgent(m)
    a = agent(m, footprints=True, filled=True)

    current_state = m.start
    while current_state != m.end:
        action = q_agent.choose_action(current_state)
        next_state, reward = a.move(action)
        q_agent.learn(current_state, action, reward, next_state)
        current_state = next_state

    m.run()

run_with_q_learning()
