import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Definir el espacio de acción: arriba, abajo, izquierda, derecha
        self.action_space = spaces.Discrete(4)

        # Definir el espacio de observación
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32)

        # Laberinto: 0 es un camino, 1 es una pared, 2 es la salida
        self.maze = np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 2],
        ])

        # Posición inicial
        self.start_pos = (0, 0)
        self.reset()

    def reset(self):
        self.position = list(self.start_pos)
        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros((5, 5))
        obs[self.position[0], self.position[1]] = 1  # Agente
        return obs

    def step(self, action):
        # Actualizar posición según la acción
        if action == 0:  # Arriba
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 1:  # Abajo
            self.position[0] = min(4, self.position[0] + 1)
        elif action == 2:  # Izquierda
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 3:  # Derecha
            self.position[1] = min(4, self.position[1] + 1)

        # Comprobar si ha alcanzado la salida
        done = False
        reward = -1  # Penalización por cada paso
        if self.maze[tuple(self.position)] == 2:
            done = True
            reward = 10  # Recompensa al llegar a la salida

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        plt.imshow(self.maze, cmap='gray')
        plt.scatter(self.position[1], self.position[0], color='red')  # Posición del agente
        plt.title("Posición del Agente")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.5)


# Crear el entorno
env = MazeEnv()

# Configuración para la animación
fig = plt.figure()
frames = []

# Ejecutar un episodio simple
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Tomar una acción aleatoria
    obs, reward, done, _ = env.step(action)  # Aplicar la acción
    frames.append(env.maze.copy())  # Guardar el estado actual del laberinto


# Crear una animación de la trayectoria del agente
def update_frame(frame):
    plt.imshow(frame, cmap='gray')
    plt.scatter(0, 0, color='red')  # Posición inicial del agente
    plt.title("Posición del Agente")
    plt.axis('off')


# Mostrar la animación
ani = animation.FuncAnimation(fig, update_frame, frames=frames, repeat=False)
plt.show()
