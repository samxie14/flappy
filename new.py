import flappy_bird_gymnasium
import gymnasium
import pygame
import torch
import torch.nn as nn


def generate_past_states(episode_transitions, state_dim):
    num_transitions = len(episode_transitions)
    num_pad = max(0, 16 - num_transitions)
    
    if num_transitions == 0:
        # If no transitions, return all zeros
        states = torch.zeros(16, state_dim)
        return states
    
    # Build list of states first
    #states_list = []
    states_list = torch.zeros(num_pad, state_dim)
    # Add padding if needed
    # for _ in range(num_pad):
    #     states_list.append(torch.zeros(state_dim))
    #states_list.append(torch.zeros(num_pad, state_dim))
    # Add actual states
    for i in range(max(0, num_transitions-16), num_transitions):
        states_list = torch.cat((states_list, episode_transitions[i][0].unsqueeze(0)), dim=0)
        #states_list.append(episode_transitions[i][0])
    if(len(episode_transitions) == 1):
        print(episode_transitions[0][0])
        print(states_list)
    # Stack all states at once
    #states = torch.stack(states_list, dim=0)


    
    #print(f"states shape: {states.shape}")
    return states_list


if __name__ == '__main__':
    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=True)
    state_dim = env.observation_space.shape[0]

    state, _ = env.reset()
    episode_transitions = []
    while True:
        # Handle pygame events for human input
        action = env.action_space.sample()
        action = torch.tensor(action, dtype=torch.int64)

        states = generate_past_states(episode_transitions, state_dim)
        state = torch.tensor(state, dtype=torch.float)

        # Take step in environment
        next_state, reward, terminated, _, info = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype =torch.float)

        episode_transitions.append((state, action, next_state, reward, terminated))
        
        if terminated:
            break
        state = next_state  

    env.close()