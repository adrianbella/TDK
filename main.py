import gym
import datetime

from model import CNN
from logger import Logger
from config import MyConfigParser

from rl.agents import DQNAgent
from rl.agents import CEMAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory

from keras.optimizers import Adam

if __name__ == "__main__":

    ENV_NAME = 'VirtualDrone-v0'
    section = 'DQNAgent'
    env = gym.make(ENV_NAME)  # environment initialization
    logger = Logger(section)
    config = MyConfigParser(section)

    lr = float(config.config_section_map()['learningrate'])

    cnn = CNN(env.action_space.n)
    logger.log_model_architect(cnn.model)

    limit = int(config.config_section_map()['memorylimit'])

    policy = GreedyQPolicy()
    nb_actions = env.action_space.n
    nb_steps = int(config.config_section_map()['iterations'])
    nb_episodes = int(config.config_section_map()['evaluationepisodes'])

    dqn_memory = SequentialMemory(limit=limit, window_length=1)
    cem_memory = EpisodeParameterMemory(limit=limit, window_length=1)

    if section == 'DQNAgent':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=False,
                         enable_dueling_network=False,
                         dueling_type='avg')
    elif section == 'DDQNAgent':
        agent = DQNAgent(model=cnn.model, policy=policy, nb_actions=nb_actions, memory=dqn_memory,
                         enable_double_dqn=True,
                         enable_dueling_network=True,
                         dueling_type='avg')
    elif section == 'CEMAgent':
        agent = CEMAgent(model=cnn.model, nb_actions=nb_actions, memory=cem_memory, batch_size=32, nb_steps_warmup=10000,
                         train_interval=100,
                         elite_frac=0.05)

    if section == 'DQNAgent' or section == 'DDQNAgent':
        agent.compile(Adam(lr=lr), metrics=['mse'])
    else:
        agent.compile()

    history = agent.fit(env, action_repetition=1, nb_steps=nb_steps, callbacks=[logger], visualize=False,
                        verbose=2)

    logger.log_history(history)

    # After training is done, we save the final weights.
    agent.save_weights(ENV_NAME + '_' + section + '_' + str(datetime.datetime.now()) + '_weights.h5f',
                       overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=nb_episodes, callbacks=[logger], visualize=False, verbose=2)

    print('end')
