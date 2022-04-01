# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import gym


def main():
    # removing render will make the training much faster
    env = gym.make('Boxing-v0', render_mode='human')
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())  # take a random action
        print(i)
    env.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
