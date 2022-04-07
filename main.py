from pettingzoo.atari import boxing_v1
from utils import *
from tqdm import tqdm
from model import DeepQNetwork

## defining the environment properties.
env = boxing_v1.parallel_env()

env = preset_settings_env(env)


def main():
    observations = env.reset()
    max_cycles = 50000
    DQN = DeepQNetwork(env, env.agents[0])

    for step in tqdm(range(max_cycles), total=max_cycles):
        prev_obs = observations
        actions = {DQN.agent_name: DQN.choose_action(observations[DQN.agent_name].transpose(2,0,1)), env.agents[1]: policy(observations, env.agents[1])}
        #actions = {agent: policy(observations[agent], agent) for agent in env.agents}
        observations, rewards, dones, infos = env.step(actions)
        ## process related to dqn agent

        DQN.store_transition(observations[DQN.agent_name].transpose(2,0,1), actions[DQN.agent_name], rewards[DQN.agent_name], prev_obs[DQN.agent_name].transpose(2,0,1), dones[DQN.agent_name])
        if len(DQN.memory)>1000 and DQN.state_counter%4 :
            loss,avg_q = DQN.learn()

        if DQN.state_counter%10000 ==0:
            DQN.target_net.load_state_dict(DQN.eval_net.state_dict())

        if len(env.agents) < 2:
            ## reset environment and play again.
            observations = env.reset()

        env.render()


def policy(obsv, agent):
    return env.action_space(agent).sample()


if __name__ == '__main__':
    main()

