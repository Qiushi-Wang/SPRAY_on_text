import torch
import numpy as np

def lrp(fc_param, conv_value, output, label_num):
    reward_fc = output
    reward_conv_fc = torch.zeros([label_num, 200])
    for i in range(len(reward_conv_fc)):
        for j in range(len(reward_conv_fc[i])):
            reward_conv_fc[i][j] = (conv_value[j] * fc_param[i][j] / sum(conv_value * fc_param[i][:])) * reward_fc[i]

    reward_conv = torch.zeros(200)
    for i in range(len(reward_conv)):
        reward_conv[i] = 0
        for j in range(len(reward_conv_fc)):
            reward_conv[i] = reward_conv[i] + reward_conv_fc[j][i]


    reward_conv1 = reward_conv[:len(reward_conv)//2] #(100,1)
    reward_conv2 = reward_conv[len(reward_conv)//2:] #(100,1)

    # conv_param1 = conv_param[0] #(1,3,100)
    # conv_param2 = conv_param[1] #(1,3,100)

    # todo: conpute the reward of input
    # use: input, reward_conv1,2 as R_j, conv_param conv input item as q_ij and input:(100, 100)
    reward_input = torch.zeros(100)
    for i in range(100):
        if i == 0:
            reward_input[0] = reward_input[0] + reward_conv1[i]/2 + reward_conv2[i]/2#0
            reward_input[1] = reward_input[1] + reward_conv1[i]/2 + reward_conv2[i]/2#1
        elif i == 99:
            reward_input[99] = reward_input[99] + reward_conv1[i]/2 + reward_conv2[i]/2#99
            reward_input[98] = reward_input[98] + reward_conv1[i]/2 + reward_conv2[i]/2#98
        else:
            reward_input[i-1] = reward_input[i-1] + reward_conv1[i]/3 + reward_conv2[i]/3
            reward_input[i] = reward_input[i] + reward_conv1[i]/3 + reward_conv2[i]/3
            reward_input[i+1] = reward_input[i+1] + reward_conv1[i]/3 + reward_conv2[i]/3
    return reward_input













