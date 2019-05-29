from model import CNN
from master_model import OriginalCNN
import gym
import numpy as np


def evaluate_model(env, AGENT, agent_directory, current_weights_number, saved_weights_finish_number):

    hidden_fc_size = 8
    hidden_conv1_filters = 8
    hidden_conv2_filters = 16

    accuracies = []
    total_avg_episode_reward_sums = []
    total_avg_step_counts = []

    while (current_weights_number <= saved_weights_finish_number):
        #file_path = "./" + agent_directory + "/VirtualDrone-v0_" + AGENT + "_" + str(current_weights_number) + ".h5f"
        #model = OriginalCNN(env.action_space.n, file_path)

        file_path = "./master_student/" + agent_directory + "/VirtualDrone-v0_" + AGENT + "_" + 'conv18_conv216_fc8_' + str(current_weights_number) + ".h5f"
        model = CNN(env.action_space.n, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters, file_path)

        observation = env.reset()

        total_step_count = 0
        total_incorrect_step_count = 0
        total_avg_episode_reward = 0

        episode_reward_sum = 0
        episode_counter = 0

        while (episode_counter < 300):

            # env.render()
            observation = np.expand_dims(np.expand_dims(observation, axis=0), axis=0)
            prediction = model.model.predict(observation, batch_size=1)

            step = np.argmax(prediction, axis=1)

            observation, reward, done, info = env.step(step)

            total_step_count += 1

            episode_reward_sum += reward

            if reward == -4:
                total_incorrect_step_count += 1

            if done:
                episode_counter += 1
                total_avg_episode_reward += episode_reward_sum

                observation = env.reset()
                episode_reward_sum = 0

        accuracies.append([1 - (total_incorrect_step_count / total_step_count), current_weights_number])
        total_avg_episode_reward_sums.append([total_avg_episode_reward / episode_counter, current_weights_number])
        total_avg_step_counts.append([total_step_count / episode_counter, current_weights_number])

        #current_weights_number += 100000
        current_weights_number += 10240

    print_result(accuracies, total_avg_episode_reward_sums, total_avg_step_counts, AGENT)


def print_result(accuracies, total_avg_episode_reward_sums, total_avg_step_counts, AGENT):

    print('-------------------------------------------------{}-------------------------------------------------'.format(AGENT))

    for i in range(0, len(accuracies)):
        print('Accuracie:{}, Average episodic reward:{}, Average episodic steps:{}'.format(accuracies[i],
                                                                                            total_avg_episode_reward_sums[i],
                                                                                            total_avg_step_counts[i]))

if __name__ == "__main__":

    ENV_NAME = 'VirtualDrone-v0'
    env = gym.make(ENV_NAME)  # environment initialization
    AGENT = 'SARSAAgent'
    agent_directory = 'SARSA'

    evaluate_model(env, 'DQNAgent', 'DQN', 0, 174080)
    evaluate_model(env, 'DDQNAgent', 'DDQN', 0, 266240)
    evaluate_model(env, 'SARSAAgent', 'SARSA', 0, 163840)

    print('-------------------------------------------------EVALUATIONS FINISHED-------------------------------------------------')