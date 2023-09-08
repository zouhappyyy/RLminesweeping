import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Action1(nn.Module): #根据当前状态输出一个动作概率分布
    #10x10的输入图像，经过一系列卷积层和非线性激活函数的处理后，输出一个概率分布，用于表示预测的动作。
    def __init__(self,input_shape=[10,10]): #

        super(Action1,self).__init__()
        self.input_dim=input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), #  第四个卷积层：输入通道数为32，输出通道数为1，卷积核大小为3x3，步长为1，填充为1。
        )
        self.softmax = nn.Softmax(dim=1)  #应用在卷积层输出上的维度1上
        self.relu = nn.ReLU() #另一个是ReLU函数，用于在模型的forward方法中使用。

    def forward(self,x):
        x=self.conv_layers(x).view(x.shape[0],-1)
        out = self.softmax(x)
        return out

class Action2(nn.Module): #Action2网络则通常采用全连接神经网络，输入为当前状态的特征向量，输出为一个动作概率分布向量
    def __init__(self,input_shape=[10,10]):
        super(Action2,self).__init__()
        self.input_dim=input_shape[0]*input_shape[1]
        self.output_dim=(input_shape[0]+6)*(input_shape[1]+6)
        self.liner=nn.Linear(self.input_dim,512) #第一个全连接层输入大小为输入特征向量的维度（input_dim），输出大小为512。
        self.liner2=nn.Linear(512,self.output_dim) #第二个全连接层：输入大小为512，输出大小为输出特征向量的维度（output_dim）。
        self.liner3 = nn.Linear(self.output_dim,self.input_dim) #输入大小为输出特征向量的维度（output_dim），输出大小为输入特征向量的维度（input_dim）。

        self.softmax = nn.Softmax(dim=1) #应用在第三个全连接层的输出上，用于产生输出的概率分布
        self.relu = nn.ReLU()

    def forward(self,x): #在forward方法中，输入x首先通过展平操作将其变成一维张量，然后经过第一个全连接层和ReLU激活函数的处理。
        # 接着，再经过第二个全连接层和ReLU激活函数的处理。最后，通过第三个全连接层和softmax激活函数的处理，得到输出的动作概率分布向量。
        x=x.view(x.shape[0],-1)
        x=self.relu(self.liner(x))
        x=self.relu(self.liner2(x))
        out=self.softmax(self.liner3(x))
        return out

class Bvalue(nn.Module): #Bvalue网络的作用是估计在当前状态下，采取某个动作所能获得的预期累积奖励。具体实现上，
        # Bvalue网络通常采用全连接神经网络，输入为当前状态的特征向量，输出为一个标量值，代表当前状态的价值。在PPO算法中，Bvalue网络的损失函数通常采用均方误差损失，目标值为当前状态下的实际累积奖励。
    def __init__(self):
        super(Bvalue,self).__init__()
        self.relu = nn.ReLU()
        self.liner=nn.Linear(200,256)
        self.liner2=nn.Linear(256,512)
        self.liner3 = nn.Linear(512,1) #线性层接收512维向量作为输入，并输出一个标量值，即当前状态的价值
        #损失函数采用均方误差损失，目标值则是当前状态下的实际累积奖励。

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x=self.relu(self.liner(x))
        x=self.relu(self.liner2(x))
        out = self.liner3(x)
        return out

class PPO():

    def __init__(self,input_shape=[10,10],up_time=10,batch_size=32,a_lr=1e-5,b_lr=1e-5,gama=0.9,epsilon=0.1):
        #初始化算法所需参数，以及两个神经网络模型 action 和 bvalue，并创建了对应的优化器和损失函数。
        self.up_time=up_time
        self.batch_size=batch_size
        self.gama=gama
        self.epsilon=epsilon
        self.suffer = []
        self.action = Action1(input_shape)
        self.action.to(device)
        self.bvalue = Bvalue()
        self.bvalue.to(device)
        self.acoptim = optim.Adam(self.action.parameters(), lr=a_lr)
        self.boptim = optim.Adam(self.bvalue.parameters(), lr=b_lr)
        self.loss = nn.MSELoss().to(device)
        self.old_prob = []

    def appdend(self, buffer):
        self.suffer.append(buffer)

    def load_net(self,path): #从预训练模型文件中加载 action 模型
        self.action=torch.load(path)

    def get_action(self, x): #接受一个状态 x，通过 action 模型计算出给定状态下各个动作的概率分布，并按照概率分布采样得到一个动作
        x = x.unsqueeze(dim=0).to(device)
        ac_prob = self.action(x)

        a = Categorical(ac_prob).sample()[0]  # 按概率采样

        # values, indices = ac_prob.topk(k=15,dim=1)
        # a = Categorical(values).sample()[0]  # 按topk15概率采样
        # a = indices[0,a]

        ac_pro = ac_prob[0][a]
        return [a.item()], [ac_pro.item()]

    def update(self): #接受一个状态 x，通过 action 模型计算出给定状态下各个动作的概率分布，并按照概率分布采样得到一个动作
        states = torch.stack([t.state for t in self.suffer],dim=0).to(device)
        actions = torch.tensor([t.ac for t in self.suffer], dtype=torch.int).to(device)
        rewards = [t.reward for t in self.suffer]
        done=[t.done for t in self.suffer]
        old_probs = torch.tensor([t.ac_prob for t in self.suffer], dtype=torch.float32).to(device)  # .detach()

        false_indexes = [i+1 for i, val in enumerate(done) if not val]
        if len(false_indexes)>=0:
            idx,reward_all=0,[]
            for i in false_indexes:
                reward=rewards[idx:i]
                R = 0
                Rs = []
                reward.reverse()
                for r in reward:
                    R = r + R * self.gama
                    Rs.append(R)
                Rs.reverse()
                reward_all.extend(Rs)
                idx=i
        else:
            R = 0
            reward_all = []
            rewards.reverse()
            for r in rewards:
                R = r + R * self.gama
                reward_all.append(R)
            reward_all.reverse()
        Rs = torch.tensor(reward_all, dtype=torch.float32).to(device)
        for _ in range(self.up_time):
            self.action.train()
            self.bvalue.train()
            for n in range(max(10, int(10 * len(self.suffer) / self.batch_size))):
                index = torch.tensor(random.sample(range(len(self.suffer)), self.batch_size), dtype=torch.int64).to(device)
                v_target = torch.index_select(Rs, dim=0, index=index).unsqueeze(dim=1)
                v = self.bvalue(torch.index_select(states, 0, index))
                adta = v_target - v
                adta = adta.detach() #adta表示在当前状态下执行动作所获得的实际奖励和当前状态价值函数的差值
                probs = self.action(torch.index_select(states, 0, index))
                pro_index = torch.index_select(actions,0,index).to(torch.int64)

                probs_a = torch.gather(probs, 1, pro_index)
                ratio = probs_a / torch.index_select(old_probs, 0, index).to(device)#ratio表示新旧策略之间的动作概率分布比值
                surr1 = ratio * adta
                surr2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * adta.to(device) #self.epsilon是超参数，用于控制策略更新的幅度
                action_loss = -torch.mean(torch.minimum(surr1, surr2)) #通过比较surr1和surr2，选择其中更小的一个，作为策略梯度损失action_loss
                self.acoptim.zero_grad() #self.acoptim 表示Actor和Critic网络的优化器
                # 通过调用zero_grad()方法清除梯度，然后调用backward()方法计算梯度，最后调用step()方法更新网络参数
                action_loss.backward(retain_graph=True)
                self.acoptim.step()

                bvalue_loss = self.loss(v_target, v) #v_target表示当前状态下的实际累积奖励，v表示Critic网络对当前状态的估计价值。
                # bvalue_loss采用均方误差损失，用于衡量Critic网络的估计值和实际值之间的差距，并通过反向传播更新Critic网络的参数
                self.boptim.zero_grad() #self.boptim也表示Actor和Critic网络的优化器
                bvalue_loss.backward()
                #通过调用zero_grad()方法清除梯度，然后调用backward()方法计算梯度，最后调用step()方法更新网络参数

                self.boptim.step()
                #由于Actor网络和Critic网络是共享的，因此需要分别对两个网络的参数进行更新。
        self.suffer = []