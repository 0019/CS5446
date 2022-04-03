from pettingzoo.atari import boxing_v1

env = boxing_v1.parallel_env()


def main():
    observations = env.reset()
    max_cycles = 500
    for step in range(max_cycles):
        actions = {agent: policy(observations[agent], agent) for agent in env.agents}
        observations, rewards, dones, infos = env.step(actions)
        env.render()


def policy(obsv, agent):
    return env.action_space(agent).sample()


if __name__ == '__main__':
    main()
