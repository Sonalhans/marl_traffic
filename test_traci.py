import traci
from sumolib import checkBinary

sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"]

traci.start(sumoCmd)

tls_ids = traci.trafficlight.getIDList()
print("Traffic Lights:", tls_ids)

for step in range(300):
    traci.simulationStep()

    for tls in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tls)
        queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

        # Get full program
        logic = traci.trafficlight.getAllProgramLogics(tls)[0]
        num_phases = len(logic.phases)

        # Change phase safely every 20 steps
        if step % 20 == 0:
            current_phase = traci.trafficlight.getPhase(tls)
            next_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(tls, next_phase)

        print(f"Step {step} | TLS {tls} | Queue: {queue}")

traci.close()