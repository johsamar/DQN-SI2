import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import layers, models

# Define model
class DQN(tf.keras.Model):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(h1_nodes, activation='relu', input_shape=(in_states,))
        self.out = layers.Dense(out_actions)
        self.num_actions = out_actions

    def call(self, x):
        x = self.fc1(x)
        return self.out(x)

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# MountainCar Deep Q-Learning
class MountainCarDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.99        # discount rate (gamma)    
    network_sync_rate = 1000      # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 100000    # size of replay memory
    mini_batch_size = 32           # size of the training data set sampled from the replay memory
    
    num_divisions = 20

    # Neural Network
    loss_fn = tf.keras.losses.MeanSquaredError()  # NN Loss function.
    optimizer = None                # NN Optimizer. Initialize later.

    # Train the environment
    def train(self, episodes, render=False):
        # Create MountainCar instance
        env = gym.make('MountainCar-v0', render_mode='human' if render else None)
        num_states = env.observation_space.shape[0] # expecting 2: position & velocity
        num_actions = env.action_space.n

        # Divide position and velocity into segments
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07
    
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=24, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=24, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())
        
        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
        goal_reached=False
        best_rewards=-300 
            
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal

            rewards = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and rewards>-1000):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=idle,2=right
                else:
                    # select best action            
                    q_values = policy_dqn(self.state_to_dqn_input(state))
                    action = tf.argmax(q_values, axis=1).numpy()[0]

                # Execute action
                new_state, reward, terminated, _, _ = env.step(action)

                # Accumulate reward
                rewards += reward

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(rewards)
            if(terminated):
                goal_reached = True

            # Graph training progress        
            print(f'Episode {i} Epsilon {epsilon} Rewards {rewards}')
            self.plot_progress(rewards_per_episode, epsilon_history, i)                

            if rewards > best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                # Save policy
                policy_dqn.save_weights(f"MountainCar/mountaincarv1/mountaincar_dql_{i}.h5")

            # Check if enough experience has been collected
            if len(memory) > self.mini_batch_size and goal_reached:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())
                    step_count = 0                    

        # Close environment
        env.close()

    def plot_progress(self, rewards_per_episode, epsilon_history,i):
        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(rewards_per_episode)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('MountainCar/mountaincarv1/mountaincar_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        state_batch = np.array([transition[0] for transition in mini_batch])
        action_batch = np.array([transition[1] for transition in mini_batch])
        new_state_batch = np.array([transition[2] for transition in mini_batch])
        reward_batch = np.array([transition[3] for transition in mini_batch])
        terminated_batch = np.array([transition[4] for transition in mini_batch])

        # Convert to tensor
        state_batch = self.state_to_dqn_input(state_batch)
        new_state_batch = self.state_to_dqn_input(new_state_batch)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        terminated_batch = tf.convert_to_tensor(terminated_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Predict Q-values for the current state
            current_q = policy_dqn(state_batch)
            current_q = tf.reduce_sum(current_q * tf.one_hot(action_batch, policy_dqn.num_actions), axis=1)

            # Predict Q-values for the next state
            next_q = target_dqn(new_state_batch)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target_q = reward_batch + self.discount_factor_g * max_next_q * (1 - terminated_batch)

            # Compute the loss
            loss = self.loss_fn(target_q, current_q)

        # Apply gradients
        grads = tape.gradient(loss, policy_dqn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, policy_dqn.trainable_weights))

    '''
    Converts a state (position, velocity) to tensor representation.
    Example:
    Input = (0.3, -0.03)
    Return = tensor([16, 6])
    '''
    def state_to_dqn_input(self, state)->tf.Tensor:
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        state_p = np.digitize(state[:, 0], self.pos_space)
        state_v = np.digitize(state[:, 1], self.vel_space)
        return tf.convert_to_tensor(np.stack([state_p, state_v], axis=1), dtype=tf.float32)
    
    # Run the environment with the learned policy
    def test(self, episodes, model_filepath):
        # Create MountainCar instance
        env = gym.make('MountainCar-v0', render_mode='human')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        policy_dqn.load_weights(model_filepath)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            
            total_rewards = []
            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                q_values = policy_dqn(self.state_to_dqn_input(np.array([state])))
                action = tf.argmax(q_values, axis=1).numpy()[0]

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f'Episode {i + 1}: Total Reward: {episode_reward}')
        env.close()
        average_reward = np.mean(total_rewards)
        print(f'Average Reward over {episodes} episodes: {average_reward}')
        return average_reward
    
if __name__ == '__main__':
    mountaincar = MountainCarDQL()
    mountaincar.train(100, False)
    # mountaincar.test(10, "MountainCar/mountaincarv1/mountaincar_dql_17000.weights.h5")
