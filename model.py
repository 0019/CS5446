import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple




class deep_net(nn.Module):
    def __init__(self, num_actions, seed=13):
        super(deep_net, self).__init__()

        self.seed = seed
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,  kernel_size=2)  # output shape 64 x 18 x 18
        
        ## defining linear layers.

        self.linear1 = nn.Linear(4096,1024)
        self.linear2 = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, num_actions)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.1)
        
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        x = x/255 ## scaling to 0, 1
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 4096)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.classifier(x)
        return x


class DeepQNetwork:
    def __init__(self, env, agent_name) -> None:
        self.agent_name = agent_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## Hyper params for the deepnet.
        self.learning_rate = 0.0000625
        self.gamma = 0.99
        self.epsilon = 1
        self.batchsize = 32
        self.Experience = namedtuple('Experience',
                        ('s', 'a', 'r', 's_','d'))
        self.env = env
        self.num_actions = self.env.action_space(self.agent_name).n
        
        ## target net and eval net
        self.target_net = deep_net(self.num_actions, seed=13).to(self.device)
        self.eval_net = deep_net(self.num_actions, 13).to(self.device)
        self.eval_net.apply(self.eval_net.init_weights)
       
        ## eval net weights are copied to target net and set to eval mode.
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.memory_counter = 0
        self.state_counter = 0
        self.memory = []
        self.optimizer = Adam(self.eval_net.parameters(),lr=self.learning_rate,eps = 1.5e-4)
        self.epsilon = 1
        self.state_size = 5 
        self.start_data_size = 50000
    
    def choose_action(self,x,train=True):
        ## epsilon greedy choice.
        if train==True:
            if len(self.memory) > self.start_data_size:
                self.epsilon -= (1-0.1)/1000000
                self.epsilon = max(self.epsilon, 0.1)
            
            eps = self.epsilon
        else:
            eps = 0.05
        
        if np.random.uniform() > eps:
            x = torch.unsqueeze(torch.tensor(np.array(x,dtype=np.float32),device=self.device,dtype=torch.float32),0).to(self.device)
            q_value = self.eval_net(x).detach()
            action = torch.argmax(q_value).item()
        
        else:
            action = self.env.action_space(self.agent_name).sample()

        return action

    def store_transition(self,s,a,r,s_,d):
        self.state_counter += 1
        exp = [s,a,r,s_,d]
        if len(self.memory) >= 1000000:
            self.memory.pop(0)
        self.memory.append(exp)
    
    def learn(self):



        sample = random.sample(self.memory , self.batchsize)
        batch = self.Experience(*zip(*sample))

        b_s = torch.tensor(np.array(batch.s,dtype=np.float32),device=self.device,dtype=torch.float32).to(self.device)
        b_a = torch.tensor(batch.a,device=self.device).unsqueeze(1).to(self.device)
        b_r = torch.tensor(np.array(batch.r,dtype=np.float32),device=self.device,dtype=torch.float32).unsqueeze(1).to(self.device)
        b_s_ = torch.tensor(np.array(batch.s_,dtype=np.float32),device=self.device,dtype=torch.float32).to(self.device)
        b_d = torch.tensor(np.array(batch.d,dtype=np.float32),device = self.device,dtype=torch.float32).unsqueeze(1).to(self.device)

        
        q_eval = torch.gather(self.eval_net(b_s),1,b_a)
        avg_q = torch.sum(q_eval.detach())/self.batchsize
        q_eval = q_eval.to(self.device)
       
        #ddqn
        #argmax = self.eva_net(b_s_).detach().max(1)[1].long()

        #q_next = self.target_net(b_s_).detach().gather(1,torch.unsqueeze(argmax,1))
        q_next = self.target_net(b_s_).detach()   #target network is not updated. 
        q_next = q_next.to(self.device)
       
        q_target = b_r + self.gamma*q_next.max(1)[0].unsqueeze(1)*(-b_d+1)

   
        #q_target = b_r + GAMMA*q_next*(-b_d+1)
   
        loss = F.mse_loss(q_eval,q_target)
        
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        return loss.item(),avg_q.item()