import flappy_bird_gymnasium
import gymnasium
from gymnasium.wrappers import RecordVideo

import os
import torch
import yaml
from model import DQN
from experience_replay import Replay
import random
import torch.nn.functional as F
import torch.optim as optim


class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            hyperparameters = yaml.safe_load(file)
            if hyperparameters:
                hyperparameter_vars = hyperparameters[hyperparameter_set]
            else:
                print("Hyperparameters is None")

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available, using CPU")
        self.N = hyperparameter_vars['N']
        self.dropout = hyperparameter_vars['dropout']
        self.seq_len = hyperparameter_vars['seq_len']
        self.num_hidden_1 = hyperparameter_vars['num_hidden_1']
        self.num_hidden_2 = hyperparameter_vars['num_hidden_2']
        self.d_model = hyperparameter_vars['d_model']
        self.d_ff = hyperparameter_vars['d_ff']
        self.heads = hyperparameter_vars['heads']
        self.num_episodes = hyperparameter_vars['num_episodes']
        self.env_id = hyperparameter_vars['env_id']
        self.network_sync_rate = hyperparameter_vars['network_sync_rate']
        self.learning_rate = hyperparameter_vars['learning_rate']
        self.maxlen = hyperparameter_vars['maxlen']
        self.batch_size = hyperparameter_vars['batch_size']
        self.discount_factor_g = hyperparameter_vars['discount_factor_g']

        self.episode_rewards = []
        self.avg_rewards = []
        self.best_reward = float('-inf')
        self.best_episode_transitions = None
        self.best_episode_num = 0
        self.best_episode_seed = None

    def generate_past_states(self, episode_transitions, state_dim):
        num_transitions = len(episode_transitions)
        num_pad = max(0, self.seq_len - num_transitions)

        if num_transitions == 0:
            return torch.zeros(self.seq_len, state_dim, device=self.device)
        
        elif num_transitions >= self.seq_len:
            states_list = []
            for i in range(num_transitions-self.seq_len, num_transitions):
                states_list.append(episode_transitions[i][0])  # Already on device
            return torch.stack(states_list, dim=0)  # No need for .to(self.device)

        else:
            states_list = torch.zeros(num_pad, state_dim, device=self.device)  # Create directly on device

            for i in range(max(0, num_transitions-self.seq_len), num_transitions):
                states_list = torch.cat((states_list, episode_transitions[i][0].unsqueeze(0)), dim=0)
            return states_list  # No need for .to(self.device) because episode_transiitions itself is already a tensor

    def run(self):
        env= gymnasium.make("FlappyBird-v0", render_mode =None, use_lidar = True)
        state_dim  = env.observation_space.shape[0]
        num_actions = env.action_space.n
        count  = 0


        policy_q_net = DQN(state_dim, num_actions, self.num_hidden_1, self.num_hidden_2, self.d_model, self.d_ff, self.heads, self.N, self.dropout, self.seq_len).to(self.device)
        target_q_net = DQN(state_dim, num_actions, self.num_hidden_1, self.num_hidden_2, self.d_model, self.d_ff, self.heads, self.N, self.dropout, self.seq_len).to(self.device)
        target_q_net.load_state_dict(policy_q_net.state_dict())
        memory = Replay(self.maxlen)
        optimizer = torch.optim.Adam(policy_q_net.parameters(), lr=self.learning_rate)

        epsilon = 1           
        epsilon_min = 0.05
        epsilon_decay = 0.995

        for episode in range(self.num_episodes):
            # Use episode number as seed to ensure reproducibility
            # This way we can replay the exact same environment configuration
            episode_seed = episode
            state, info = env.reset(seed=episode_seed)
            episode_reward = 0

            state = torch.tensor(state, dtype=torch.float).to(self.device)
            episode_transitions = []
            while True:
                # Get state sequence for transformer 
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=self.device)
                else: 
                    states = self.generate_past_states(episode_transitions, state_dim)
                    with torch.no_grad():
                        q_values = policy_q_net(states)
                        action = torch.argmax(q_values, dim=1).squeeze(0)
                # Get Q-values and select action
                
                next_state, reward, done, _, info = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
                reward = torch.tensor(reward, dtype =torch.float).to(self.device)
                episode_transitions.append((state, action, next_state, reward, done))

                episode_reward += reward
                if done:
                    episode_reward_float = float(episode_reward.item() if torch.is_tensor(episode_reward) else episode_reward)
                    self.episode_rewards.append(episode_reward_float)
                    self.avg_rewards.append(sum(self.episode_rewards)/len(self.episode_rewards))
                    
                    # Track best episode
                    if episode_reward_float > self.best_reward:
                        self.best_reward = episode_reward_float
                        self.best_episode_transitions = episode_transitions.copy()
                        self.best_episode_num = episode + 1
                        self.best_episode_seed = episode_seed
                        print(f"Episode {episode+1} - Reward: {episode_reward_float:.2f} - Average Reward: {self.avg_rewards[-1]:.2f} NEW BEST!")
                    else:
                        print(f"Episode {episode+1} - Reward: {episode_reward_float:.2f} - Average Reward: {self.avg_rewards[-1]:.2f}")
                    break

                state = next_state
            
            memory.push(episode_transitions)

            count += 1

            if memory.size > self.batch_size:

                batch = memory.sample(self.batch_size, state_dim, self.seq_len, self.device)
                self.optimize(batch, policy_q_net, target_q_net, optimizer)
                #start training
                if count > self.network_sync_rate:
                    target_q_net.load_state_dict(policy_q_net.state_dict())
                    count = 0
            
            epsilon = max(epsilon_min, epsilon*epsilon_decay)
        
        torch.save(policy_q_net.state_dict(), "trained_model.pth")
        env.close()
        
        # Record best episode as video
        if self.best_episode_transitions is not None:
            print(f"\nRecording best episode (Episode {self.best_episode_num}, Reward: {self.best_reward:.2f}) as video...")
            self.record_best_episode(policy_q_net, state_dim, num_actions)
        
    
    def optimize(self, batch, policy_q_net, target_q_net, optimizer):
        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        states_next = batch[3]
        dones = batch[4]

        with torch.no_grad():
            next_actions = policy_q_net(states_next).argmax(dim=1)
            target_q = target_q_net(states_next).gather(1, next_actions.unsqueeze(1)).squeeze(1) #.gather preserves the shape of the index tensor
            new_q = rewards + (1-dones)*(target_q * self.discount_factor_g)
        
        policy_q = policy_q_net(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(policy_q, new_q.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def record_best_episode(self, policy_q_net, state_dim, num_actions):
        """Record the best episode as an MP4 video using the trained model"""
        
        # Create video directory
        video_dir = "videos"
        os.makedirs(video_dir, exist_ok=True)
        
        # Create environment with video recording
        env = gymnasium.make(self.env_id, render_mode="rgb_array", use_lidar=True)
        env = RecordVideo(env, video_dir, name_prefix=f"best_episode_{self.best_episode_num}")
        
        # Set model to eval mode
        policy_q_net.eval()
        
        # Reset environment with the SAME seed as the best episode
        # This ensures the pillar positions are identical to the original episode
        state, _ = env.reset(seed=self.best_episode_seed)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        # Replay using the model's policy (deterministic, no exploration)
        episode_transitions = []
        total_reward = 0
        
        with torch.no_grad():
            while True:
                # Build state sequence
                states = self.generate_past_states(episode_transitions, state_dim)
                states = states.unsqueeze(0)  # Add batch dimension: (1, seq_len, state_dim)
                
                # Get action from model
                q_values = policy_q_net(states)
                action = torch.argmax(q_values, dim=1).squeeze(0)
                
                # Take step in environment (this will be recorded)
                next_state, reward, done, _, info = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
                reward = torch.tensor(reward, dtype=torch.float).to(self.device)
                
                episode_transitions.append((state, action, next_state, reward, done))
                total_reward += float(reward.item() if torch.is_tensor(reward) else reward)
                
                if done:
                    break
                
                state = next_state
        
        env.close()
        policy_q_net.train()  # Set back to training mode
        
        # Find the video file
        video_files = [f for f in os.listdir(video_dir) if f.startswith(f"best_episode_{self.best_episode_num}") and f.endswith(".mp4")]
        if video_files:
            print(f"Best episode video saved to: {video_dir}/{video_files[0]}")
            print(f"   Replay reward: {total_reward:.2f} (Original: {self.best_reward:.2f})")
        else:
            print(f"Video recording completed, check {video_dir}/ for output files")


if __name__ == "__main__":
    agent = Agent("flappybird1")
    agent.run()
    
    
