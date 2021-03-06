import gym
import datetime

from model import CNN
from minimized_model import MinimizedCNN
from logger import Logger
from config import MyConfigParser

from rl.agents import DQNAgent, SARSAAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory

import numpy as np

from keras.optimizers import Adam

from tensorflow.python.keras.callbacks import TensorBoard

if __name__ == "__main__":

    ENV_NAME = 'VirtualDrone-v0'
    AGENT_TYPE = 'DDQNAgent'
    env = gym.make(ENV_NAME)  # environment initialization
    logger = Logger(AGENT_TYPE, ENV_NAME)
    config = MyConfigParser(AGENT_TYPE)
    #file_path = section + '.h5f'

    lr = float(config.config_section_map()['learningrate'])

    if AGENT_TYPE == 'minimizedDQN':
        hidden_fc_size = 8
        hidden_conv1_filters = 8
        hidden_conv2_filters = 16
        file_path = ENV_NAME + '_' + AGENT_TYPE + '_conv1' + str(hidden_conv1_filters) + '_conv2' + str(
            hidden_conv2_filters) + '_fc' + str(hidden_fc_size) + '.h5f'
        cnn = MinimizedCNN(env.action_space.n, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters)
    else:
        cnn = CNN(env.action_space.n)
    logger.log_model_architect(cnn.model)
    logger.set_model(cnn.model)

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

    if AGENT_TYPE == 'DQNAgent' or AGENT_TYPE == 'minimizedDQN':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=False,
                         enable_dueling_network=False,
                         dueling_type='max')
        #agent.load_weights(file_path)
    elif AGENT_TYPE == 'DDQNAgent':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=True,
                         enable_dueling_network=False,
                         dueling_type='max')
        #agent.load_weights(file_path)
    elif AGENT_TYPE == 'SARSAAgent':
        agent = SARSAAgent(model=cnn.model, nb_actions=nb_actions, policy=policy, test_policy=None, gamma=0.99,
                           nb_steps_warmup=1000, train_interval=1, delta_clip=np.inf)
        #agent.load_weights(file_path)

    agent.compile(Adam(lr=lr), metrics=['mse'])

    history = agent.fit(env, action_repetition=1, nb_steps=nb_steps, callbacks=[logger], visualize=False,
                        verbose=2)

    logger.log_history(history)

    # After training is done, we save the final weights.
    agent.save_weights(ENV_NAME + '_' + AGENT_TYPE + '_' + str(datetime.datetime.now()) + '_weights.h5f',
                       overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=nb_episodes, callbacks=[logger], visualize=False, verbose=2) # TODO: testpolicy-ket átgondolni!

    print('end')
