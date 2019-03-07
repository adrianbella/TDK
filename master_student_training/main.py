import gym

from master_model import MasterCNN
from student_model import StudentCNN
from logger import Logger

from agent import Agent

if __name__ == '__main__':

    ENV_NAME = 'VirtualDrone-v0'
    AGENT_TYPE = 'DQNAgent'
    database_limit = 1024
    hidden_fc_size = 256
    hidden_conv1_filters = 8
    hidden_conv2_filters = 16
    learning_rate = 0.0005
    env = gym.make(ENV_NAME)  # environment initialization
    config = ENV_NAME + '_' + AGENT_TYPE + '_conv1' + str(hidden_conv1_filters) + '_conv2' + str(hidden_conv2_filters) + '_fc' + str(hidden_fc_size)

    master_file_path = './master_weights/' + ENV_NAME + '_' + AGENT_TYPE + '.h5f'

    master_model = MasterCNN(env.action_space.n, master_file_path)
    student_model = StudentCNN(env.action_space.n, hidden_fc_size, learning_rate, hidden_conv1_filters, hidden_conv2_filters)

    logger = Logger(config, student_model)

    agent = Agent(env.action_space.n, database_limit, config)

    agent.fit_student(master_model, student_model, env, logger)

    print('end')
