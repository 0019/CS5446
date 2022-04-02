from pettingzoo.atari import boxing_v1


def main():
    env = boxing_v1.env()
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        if agent == "first_0":
            env.step(env.action_space(agent).sample())
        else:
            env.step(0)
        env.render()


if __name__ == '__main__':
    main()
