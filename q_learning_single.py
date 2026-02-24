import traci
from sumolib import checkBinary
import numpy as np
import random

alpha = 0.1
gamma = 0.9
epsilon = 0.1

Q = np.zeros((21, 2))

sumoBinary = checkBinary("sumo")

for episode in range(30):

    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"]
    traci.start(sumoCmd)

    tls_id = "B1"
    total_reward = 0

    for step in range(500):

        traci.simulationStep()

        lanes = traci.trafficlight.getControlledLanes(tls_id)
        queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

        state = min(queue // 5, 20)

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])

        if action == 1:
            current_phase = traci.trafficlight.getPhase(tls_id)
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(logic.phases)
            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % num_phases)

        traci.simulationStep()

        lanes = traci.trafficlight.getControlledLanes(tls_id)
        new_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

        next_state = min(new_queue // 5, 20)
        reward = -new_queue
        total_reward += reward

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

    traci.close()
    print(f"Episode {episode} | Total Reward: {total_reward}")