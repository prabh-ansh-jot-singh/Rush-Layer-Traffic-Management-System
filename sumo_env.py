# sumo_env.py
import os
import sys
import numpy as np

# ensure SUMO python tools are importable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation")

import traci

class SumoEnvironment:
    """
    Minimal SUMO wrapper compatible with your GNN_DQNAgent.
    - On reset() it starts SUMO (sumo or sumo-gui) with the provided .sumocfg.
    - step(action) accepts either:
         * an int -> same action applied to all TLS
         * a list/np.array of length num_nodes -> per-TLS actions
    - returns state as (features, edge_index) where:
         features: np.array shape [num_nodes, num_node_features]
         edge_index: np.array shape [2, num_edges] (dtype int)
    """
    def __init__(self, sumo_cfg, sumo_binary="sumo", num_node_features=3,
                 num_actions=3, max_steps=200, fixed_num_nodes=None, topology="fully_connected"):
        self.sumo_cfg = sumo_cfg
        self.sumo_binary = sumo_binary  # "sumo" or "sumo-gui"
        self.num_node_features = num_node_features
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.step_count = 0
        self.tls_ids = None
        self.fixed_num_nodes = fixed_num_nodes  # if you want to limit/pad nodes
        self.topology = topology
        self.prev_total_wait = None

    def _start_sumo(self):
        cmd = [self.sumo_binary, "-c", self.sumo_cfg, "--step-length", "1", "--no-step-log", "true"]
        traci.start(cmd)

    def reset(self):
        # close if open
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        self._start_sumo()
        self.step_count = 0
        # determine traffic light ids
        self.tls_ids = traci.trafficlight.getIDList()
        if not self.tls_ids:
            # If no traffic lights, you may want to treat each incoming edge as a node
            raise RuntimeError("No traffic lights found in SUMO network. Check net.sumocfg / your network.")
        # if fixed_num_nodes specified, limit or pad
        if self.fixed_num_nodes:
            if len(self.tls_ids) >= self.fixed_num_nodes:
                self.tls_ids = self.tls_ids[:self.fixed_num_nodes]
            else:
                # pad by repeating last id (simple fallback)
                while len(self.tls_ids) < self.fixed_num_nodes:
                    self.tls_ids.append(self.tls_ids[-1])

        # initial state and prev_total_wait
        state = self._get_state()
        self.prev_total_wait = self._total_waiting()
        return state

    def step(self, action):
        """
        action: int (apply to all TLS) or list/np.array of ints (per TLS)
        """
        # map action to each traffic light
        if isinstance(action, (list, tuple, np.ndarray)):
            # per-tls actions expected length == number of tls
            assert len(action) >= len(self.tls_ids), "action list length < number of TLS"
            actions = action
        else:
            # single action -> apply to all
            actions = [int(action)] * len(self.tls_ids)

        # apply actions
        for tls_id, a in zip(self.tls_ids, actions):
            # clipped to available phases for safety:
            try:
                n_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].state)
            except Exception:
                # fallback: assume at least 1 phase
                n_phases = self.num_actions
            phase = int(np.clip(int(a), 0, max(0, n_phases - 1)))
            try:
                traci.trafficlight.setPhase(tls_id, phase)
            except Exception:
                # fallback: set program and phase index if above fails
                try:
                    traci.trafficlight.setPhase(tls_id, phase)
                except Exception:
                    pass

        # step simulation by one second (or step-length defined in sumo config)
        traci.simulationStep()
        self.step_count += 1

        # new state
        next_state = self._get_state()

        # reward: encourage reduction in total waiting vehicles
        current_wait = self._total_waiting()
        reward = self.prev_total_wait - current_wait  # positive if waiting decreased
        self.prev_total_wait = current_wait

        done = (self.step_count >= self.max_steps) or (traci.simulation.getMinExpectedNumber() == 0)

        return next_state, float(reward), done

    def _get_state(self):
        """
        Build node features for each TLS:
        features per node = [total_halting_vehicles, total_vehicle_count, avg_speed]
        """
        features = []
        for tls_id in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            # dedupe lanes
            lanes = list(dict.fromkeys(lanes))
            halting = 0
            count = 0
            speed_sum = 0.0
            if not lanes:
                # zero feature if no lanes
                features.append([0.0, 0.0, 0.0])
                continue
            for lane in lanes:
                try:
                    halting += traci.lane.getLastStepHaltingNumber(lane)
                    count += traci.lane.getLastStepVehicleNumber(lane)
                    speed_sum += traci.lane.getLastStepMeanSpeed(lane)
                except Exception:
                    # lane might be unknown (if pad etc.), ignore
                    pass
            avg_speed = speed_sum / len(lanes) if len(lanes) > 0 else 0.0
            features.append([float(halting), float(count), float(avg_speed)])

        features = np.array(features, dtype=np.float32)  # shape [num_nodes, 3]

        # create an edge_index for GNN
        n = len(self.tls_ids)
        if self.topology == "fully_connected":
            src = []
            dst = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = np.array([src, dst], dtype=np.int64)
        elif self.topology == "chain":
            src = [i for i in range(n-1) for _ in (0,)]
            dst = [i+1 for i in range(n-1) for _ in (0,)]
            edge_index = np.array([src, dst], dtype=np.int64)
        else:
            # default fallback: fully connected
            src = []
            dst = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(i); dst.append(j)
            edge_index = np.array([src, dst], dtype=np.int64)

        return (features, edge_index)

    def _total_waiting(self):
        """Return total halting vehicles across all lanes in network (fast aggregate)."""
        total = 0
        try:
            for edge in traci.edge.getIDList():
                try:
                    total += traci.edge.getLastStepHaltingNumber(edge)
                except Exception:
                    pass
        except Exception:
            total = 0
        return total

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
