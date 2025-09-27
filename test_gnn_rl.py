import json
import asyncio
import websockets
import numpy as np
import torch
from gnn_rl_agent import GNN_DQNAgent
from sumo_env import SumoEnvironment

# WebSocket config
WS_HOST = "localhost"
WS_PORT = 8765  # make sure Hoppscotch uses the same


async def agent_loop(websocket):
    """
    Handles a WebSocket client connection and streams actions step-by-step.
    Runs SUMO episodes in an infinite loop until client disconnects.
    Also streams JSON messages to Hoppscotch continuously.
    """
    sumo_cfg = "data/net.sumocfg"

    # must match presentation config (4 phases)
    agent = GNN_DQNAgent(num_node_features=3, hidden_dim=64, num_actions=4)

    # ⚠️ Try loading old checkpoint (trained with 3 actions)
    try:
        state_dict = torch.load("data/gnn_dqn.pth", map_location=agent.device)
        agent.model.load_state_dict(state_dict, strict=False)  # ignore size mismatch
        print("⚠️ Loaded existing model with strict=False (trained on 3 actions)")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint, using fresh model: {e}")

    episode_num = 0

    while True:  # keep running episodes until client disconnects
        env = SumoEnvironment(
            sumo_cfg=sumo_cfg,
            sumo_binary="sumo-gui",   # requires SUMO installed + PATH set
            num_node_features=3,
            num_actions=4
        )

        state = env.reset()
        total_reward = 0.0
        done = False
        episode_data = []

        while not done:
            state_x, edge_index = state

            # ❌ Old (always 0)
            # actions = agent.act(state_x, edge_index, epsilon=0.0)

            # ✅ New: add randomness so TLS actually switches
            if np.random.rand() < 0.3:  # 30% chance pick random phase
                actions = np.random.randint(0, 4, size=state_x.shape[0])
            else:
                actions = agent.act(state_x, edge_index, epsilon=0.0)

            # JSON message
            message = {
                "actions": actions.tolist(),
                "step": env.step_count,
                "status": "running"
            }

            # Send over WebSocket
            try:
                await websocket.send(json.dumps(message))
                print(f"📤 Sent: {message}")
            except websockets.ConnectionClosed:
                print("❌ Client disconnected")
                env.close()
                return

            # Save locally
            episode_data.append(message)

            # Step SUMO
            next_state, reward, done = env.step(actions)
            state = next_state
            total_reward += reward

            # small delay for smooth streaming
            await asyncio.sleep(0.5)

        # ✅ Save episode JSON
        episode_num += 1
        filename = f"episode_{episode_num}.json"
        with open(filename, "w") as f:
            json.dump(episode_data, f, indent=4)
        print(f"💾 Saved episode data to {filename}")

        print(f"✅ Episode finished | Total Reward: {total_reward:.2f}")
        env.close()


async def main():
    async with websockets.serve(agent_loop, WS_HOST, WS_PORT):
        print(f"🚦 WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
