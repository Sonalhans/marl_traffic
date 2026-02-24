import traci
from sumolib import checkBinary
import numpy as np
import random
import time

# ===============================
# Hyperparameters
# ===============================
alpha = 0.1
gamma = 0.9
epsilon = 0.1

episodes = 5
steps_per_episode = 1000

# Use GUI (change to "sumo" if needed)
sumoBinary = checkBinary("sumo-gui")

Q_tables = None

# ===============================
# Training Loop
# ===============================

for episode in range(episodes):

    print(f"\n===== Episode {episode} =====")

    sumoCmd = [
        sumoBinary,
        "-c", "simulation.sumocfg",
        "--start",
        "--delay", "100"
    ]

    traci.start(sumoCmd)
    time.sleep(1)

    tls_ids = traci.trafficlight.getIDList()

    if Q_tables is None:
        Q_tables = {tls_id: np.zeros((21, 2)) for tls_id in tls_ids}

    total_reward = 0

    for step in range(steps_per_episode):

        traci.simulationStep()

        for tls_id in tls_ids:

            lanes = traci.trafficlight.getControlledLanes(tls_id)
            queue = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in lanes
            )

            state = min(queue // 5, 20)

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = np.argmax(Q_tables[tls_id][state])

            # Switch phase
            if action == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(logic.phases)
                traci.trafficlight.setPhase(
                    tls_id,
                    (current_phase + 1) % num_phases
                )

            traci.simulationStep()

            lanes = traci.trafficlight.getControlledLanes(tls_id)
            new_queue = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in lanes
            )

            next_state = min(new_queue // 5, 20)
            reward = -new_queue
            total_reward += reward

            Q_tables[tls_id][state, action] += alpha * (
                reward
                + gamma * np.max(Q_tables[tls_id][next_state])
                - Q_tables[tls_id][state, action]
            )

        if step % 200 == 0:
            print("Step:", step)

    traci.close()
    print(f"Episode {episode} | Total Reward: {total_reward}")

print("\nTraining Finished.")