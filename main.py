import gym
import datetime

from model import CNN
from logger import Logger
from config import MyConfigParser

from rl.agents import DQNAgent, CEMAgent, SARSAAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory

import numpy as np

from keras.optimizers import Adam

if __name__ == "__main__":

    ENV_NAME = 'VirtualDrone-v0'
    section = 'SARSAAgent'
    env = gym.make(ENV_NAME)  # environment initialization
    logger = Logger(section, ENV_NAME)
    config = MyConfigParser(section)

    lr = float(config.config_section_map()['learningrate'])
    print(env.action_space.n)
    cnn = CNN(env.action_space.n)
    logger.log_model_architect(cnn.model)

    limit = int(config.config_section_map()['memorylimit'])
    eps = float(config.config_section_map()['epsilone'])
    eps_min = float(config.config_section_map()['epsilonemin'])
    eps_dec_steps = float(config.config_section_map()['decreasesteps'])

    inner_policy = EpsGreedyQPolicy(eps=eps)
    policy = LinearAnnealedPolicy(inner_policy=inner_policy, attr='eps', value_max=eps, value_min=eps_min,
                                  nb_steps=eps_dec_steps, value_test=0)
    nb_actions = env.action_space.n
    nb_steps = int(config.config_section_map()['iterations'])
    nb_episodes = int(config.config_section_map()['evaluationepisodes'])

    dqn_memory = SequentialMemory(limit=limit, window_length=1)
    cem_memory = EpisodeParameterMemory(limit=limit, window_length=1)
    sarsa_memory = SequentialMemory(limit=limit, window_length=1)

    if section == 'DQNAgent':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=False,
                         enable_dueling_network=False,
                         dueling_type='max')
    elif section == 'DDQNAgent':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=True,
                         enable_dueling_network=True,
                         dueling_type='max')
    elif section == 'CEMAgent':
        agent = CEMAgent(model=cnn.model, nb_actions=nb_actions, memory=cem_memory, batch_size=50,
                         nb_steps_warmup=1000,
                         train_interval=1,
                         elite_frac=0.1)
    elif section == 'SARSAAgent':
        agent = SARSAAgent(model=cnn.model, nb_actions=nb_actions, policy=policy, test_policy=None, gamma=0.99,
                           nb_steps_warmup=1000, train_interval=1, delta_clip=np.inf)

    if section == 'DQNAgent' or section == 'DDQNAgent' or section == 'SARSAAgent':
        agent.compile(Adam(lr=lr), metrics=['mse'])
    else:
        agent.compile()

    print(agent.metrics_names)
    history = agent.fit(env, action_repetition=1, nb_steps=nb_steps, callbacks=[logger], visualize=False,
                        verbose=2)

    logger.log_history(history)

    # After training is done, we save the final weights.
    agent.save_weights(ENV_NAME + '_' + section + '_' + str(datetime.datetime.now()) + '_weights.h5f',
                       overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=nb_episodes, callbacks=[logger], visualize=False, verbose=2) # TODO: testpolicy-ket átgondolni!

    print('end')
