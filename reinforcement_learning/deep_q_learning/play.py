#!/usr/bin/env python3
"""Script that displays a game played by the trained DQN agent."""
import numpy as np
import gymnasium as gym
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Permute
)
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class CompatibilityWrapper(gym.Wrapper):
    """Wraps a gymnasium env to expose the old gym API for keras-rl."""

    def reset(self, **kwargs):
        """Resets the environment and returns only the observation."""
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Steps the environment using the old 4-tuple return format."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


class AtariProcessor(Processor):
    """Processes Atari observations, rewards, and states."""

    def process_observation(self, observation):
        """Converts an observation to a grayscale 84x84 uint8 image."""
        from PIL import Image
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        processed = np.array(img)
        return processed.astype('uint8')

    def process_state_batch(self, batch):
        """Normalizes a batch of states to the range [0, 1]."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clips rewards to the range [-1, 1]."""
        return np.clip(reward, -1.0, 1.0)


def build_model(window, shape, actions):
    """Builds the convolutional Q-network.

    Args:
        window: the number of stacked frames.
        shape: the shape of a single processed frame.
        actions: the number of possible actions.

    Returns:
        The Keras model.
    """
    model = keras.Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = CompatibilityWrapper(env)
    nb_actions = env.action_space.n

    window = 4
    input_shape = (84, 84)

    model = build_model(window, input_shape, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=10, visualize=True)
    env.close()
