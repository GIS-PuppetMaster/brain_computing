# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:26:09 2019

@author: Administrator
"""



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import random
import pymysql
import json
import time


"""
函数名：InitState()
函数功能：从文件读取初始的state，格式处理后作为返回值
"""
def InitState():
    stream = open("env_data.txt","r+")
    string = stream.read(100)
    state = []
    """
    处理string,将其进行修改成数字数组的格式
    并将其复制给state
    """
    return state



"""
函数名：DecideDevelopOrExplore
函数功能：决定是探索新的action，还是开发已有的action
返回值：返回1则explore，返回0则develop
"""
def DecideDevelopOrExplore():
    np.set_printoptions(precision = 2)
    rand_sign = random.rand()
    #设置epsilon（e），e > 0 and e < 1;
    #e用来控制探索还是开发，两种方法的概率各占一半
    #用取随机数的方法实现
    if rand_sign >= 0 and rand_sign < 0.5: 
        return 1
    return 0



"""
函数名：EnvTest(state,action)
函数功能：从环境获得reward和next_state
"""
def EnvTest(state,action_this,action):
    file_read_rewardandnextstate = open('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json',"r+")
    string = json.load(file_read_rewardandnextstate)
    file_read_rewardandnextstate.close()
    #open the file
    #write datas into the file
    #waiting for the environment to return reward and next_state
    timestamp = os.stat('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json').st_mtime
    file_write_stateaction = open('D:/brain_computing/QLearingEnvironment/src/main/resources/trainingState.json',"r+")
    write_dict = {"blood1":state[0],"sol1":state[1],"act1":action_this,"dis1":state[3],"blood2":state[4],"sol2":state[5],"act2":action,"dis2":state[7],"Time":time.time() * 1000000}
    json.dump(write_dict,file_write_stateaction)
    file_write_stateaction.close()
    while 1 > 0:
        if timestamp != os.stat('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json').st_mtime:
            timestamp = os.stat('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json').st_mtime
            break
    #receive the feedback of environment
    file_read_rewardandnextstate = open('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json',"r+")
    string = json.load(file_read_rewardandnextstate)
    print("nextState load over!")
    reward = string["reward1"]
    #构造next_state
    next_state = [0 for i in range(8)]
    next_state[0] = string["blood1"]
    next_state[1] = string["sol1"]
    next_state[2] = string["act1"]
    next_state[3] = string["dis1"]
    next_state[4] = string["blood2"]
    next_state[5] = string["sol2"]
    next_state[6] = string["act2"]
    next_state[7] = string["dis2"]
    file_read_rewardandnextstate.close()
    return reward,next_state



"""
函数名：TupleIntoExperience(state,action,reward,next_state)
函数功能：组织元组tuple,并将tuple加入到经验池中
"""
def TupleIntoExperience(state,action,reward,next_state):
    conn = pymysql.connect(host='localhost',user='root',passwd='997341',db='brain_computing')
    cursor = conn.cursor()
    sql = "insert into state_action_reward_nextstate (state,action,reward,nextstate) values ('%s',%f,%f,'%s')"%(state,action,reward,next_state)
    cursor.execute(sql)
    cursor.connection.commit()
    cursor.close()
    conn.close()
    return 



"""
函数名：Targetnet(state,action)
函数功能：接受agent的state和action，根据当前的dqn参数计算并返回Q值
"""
def Targetnet(state,action):
    tf.reset_default_graph()
    #构造计算数据
    state_action = np.array([[0,0,0,0,0,0,0,0,0]])
    for i in range(8):
        state_action[0][i] = state[i]
    state_action[0][8] = action
    #tf.reset_default_graph()
    #定义神经网络的输入
    StateAction = tf.placeholder(tf.float32,[None,9])
    #label = tf.placeholder(tf.float32,[None,1])
    #定义神经网络参数
    w01 = tf.Variable(tf.random_normal([9,8],stddev = 1,mean = 0,seed = 1),name = 'w01')
    bias1 = tf.Variable(tf.zeros([1,8]),name = 'bias1')
    w12 = tf.Variable(tf.random_normal([8,8],stddev = 1,mean = 0,seed = 1),name = 'w12')
    bias2 = tf.Variable(tf.zeros([1,8]),name = 'bias2')
    w23 = tf.Variable(tf.random_normal([8,1],stddev = 1,mean = 0,seed = 1),name = 'w23')
    bias3 = tf.Variable(tf.zeros([1,1]),name = 'bias3')
    #定义前向传播过程
    layer1 = tf.sigmoid(tf.matmul(StateAction,w01))
    layer2 = tf.sigmoid(tf.matmul(layer1,w12))
    Q_value = tf.sigmoid(tf.matmul(layer2,w23))
    #loss = tf.reduce_mean(tf.square(Q_value - label))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #建立会话，计算数据,
    sess = tf.Session()
    saver=tf.train.Saver([w01,w12,w23,bias1,bias2,bias3])
    #saver = tf.train.import_meta_graph('G:/python_code/dqn_weight/-30.meta')
    #model_file=tf.train.latest_checkpoint("D:/brain_computing/dqn_weight/model.ckpt")
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'D:/brain_computing/dqn_weight/model.ckpt')
    #saver.save(sess,'D:/brain_computing/dqn_weight/model.ckpt')
    lis = sess.run(Q_value,feed_dict = {StateAction:state_action})
    sess.close()
    return lis[0][0]



"""
函数名：MakeLabel(state,action)
函数功能：计算每个state和action对应的标签
"""
def MakeLabel(state,action,reward,next_state):
    if next_state[0] == 0:#这里需要修改一下下标，找到血量对应的元素
        label = reward
        return label
    maxQ_value = 0
    for i in range(9):
        maxQ_value = max(maxQ_value,Targetnet(next_state,i))
    γ = 0.5 #这里需要修改一下下标，找到比较合适的γ的值
    label = float(reward) + γ * maxQ_value
    return label


    
"""
函数名：Mainnet()
函数功能：从数据库中抽取数据，训练神经网络，将训练结果保存在相同根目录下的“dqn_weight”中
"""
def Mainnet(train_StateAction,train_label):
    tf.reset_default_graph()
    #定义神经网络结构
    StateAction = tf.placeholder(tf.float32,[None,9])
    label = tf.placeholder(tf.float32,[None,1])
    w01 = tf.Variable(tf.random_normal([9,8],stddev = 1,mean = 0,seed = 1),name = 'w01')
    bias1 = tf.Variable(tf.zeros([1,8]),name = 'bias1')
    w12 = tf.Variable(tf.random_normal([8,8],stddev = 1,mean = 0,seed = 1),name = 'w12')
    bias2 = tf.Variable(tf.zeros([1,8]),name = 'bias2')
    w23 = tf.Variable(tf.random_normal([8,1],stddev = 1,mean = 0,seed = 1),name = 'w23')
    bias3 = tf.Variable(tf.zeros([1,1]),name = 'bias3')
    #定义前向传播过程
    layer1 = tf.sigmoid(tf.matmul(StateAction,w01))
    layer2 = tf.sigmoid(tf.matmul(layer1,w12))
    Q_value = tf.sigmoid(tf.matmul(layer2,w23))
    #定义损失函数和反向传播过程
    loss = tf.reduce_mean(tf.square(Q_value - label))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    #建立会话，进行训练
    sess = tf.Session()
    #init_op = tf.global_variables_initializer()
    #sess.run(init_op)
    saver=tf.train.Saver([w01,w12,w23,bias1,bias2,bias3],max_to_keep=1)#用saver的save方法进行保存神经网络参数
    saver.restore(sess,'D:/brain_computing/dqn_weight/model.ckpt')
    print("mainnet读入上次训练结果！")
    BatchSize = 20
    for i in range(300):
        if (i + 1) % 100 == 0:
            print("training round%d"%(i + 1))
        start = i % (300 - BatchSize)
        end = start + BatchSize
        sess.run(train_step,feed_dict = {StateAction:train_StateAction[start:end],label:train_label[start:end]})
        saver.save(sess,'D:/brain_computing/dqn_weight/model.ckpt')
    print("神经网络的参数已经保存！")
    print('在这次训练接受之后，神经网络的部分参数如下：')
    print(sess.run(w01))
    sess.close()



"""
函数名：PreprocessTrainString(temp_string)
函数功能：处理从数据库中调出来的字符串，返回一个只有数据的标准字符串
"""
def PreprocessTrainString(temp_string):
    temp_string = list(temp_string)
    while '[' in temp_string:
        temp_string.remove('[')
    while ']' in temp_string:
        temp_string.remove(']')
    while ',' in temp_string:
        temp_string.remove(',')
    while ' ' in temp_string:
        temp_string.remove(' ')
    while '-' in temp_string:
        temp_string.remove('-')
    for i in range(8):
        tmp = temp_string[i]
        temp_string[i] = float(tmp)
    return temp_string   



'''
函数名：ProduceData(cnt)
函数功能：生成一波初始数据（s,a,r,s'）
'''
def ProduceData(cnt):
    for i in range(cnt):
        state = [np.random.randint(1,9),np.random.randint(0,2),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(1,9),np.random.randint(0,2),np.random.randint(0,9),np.random.randint(0,9)]
        action = np.random.randint(0,9)
        reward_nextstate = EnvTest(state,action,np.random.randint(0,9))
        reward = reward_nextstate[0]
        next_state = reward_nextstate[1]
        print(state)
        print(action)
        TupleIntoExperience(state,action,reward,next_state)
       
        
'''
为神经网络的训练构造训练数据和标签
它们原来存在于数据库中，
现在要将他们加载出来
'''
def GetDataFromDb():
    #获取用于训练神经网络的数据和标签
    #首先是从数据库中将数据抽取出来
    conn = pymysql.connect(host='localhost',user='root',passwd='997341',db='brain_computing')
    cursor = conn.cursor()
    sql = 'select * from state_action_reward_nextstate'
    cursor.execute(sql)
    train_string = cursor.fetchall()
    sql = 'select count(*) from state_action_reward_nextstate'
    cursor.execute(sql)
    cnt_string = cursor.fetchone()
    item_number = cnt_string[0]
    cursor.connection.commit()
    cursor.close()
    conn.close()
    #处理train_string，得到标准的神经网络的输入和标签
    train_StateAction = [[0 for i in range(9)] for i in range(300)]
    train_label = [[0] for i in range(300)]
    for i in range(300):
        if i % 10 == 0:
            print("make label round%d"%(i + 1))
        i = (i) % item_number
        temp_string = train_string[random.randint(0,item_number - 1)][0]
        temp_string = PreprocessTrainString(temp_string)
        for j in range(8):
            train_StateAction[i][j] = temp_string[j]
        train_StateAction[i][8] = float(train_string[i][1])
        train_label[i][0] = MakeLabel(PreprocessTrainString(train_string[i][0]),float(train_string[i][1]),float(train_string[i][2]),PreprocessTrainString(train_string[i][3]))
    print("输入、标签处理完毕！")
    return train_StateAction,train_label

'''
如果想要进一步地训练神经网络，
就进一步地运行这个函数
'''
def TrainFunction():
    string = GetDataFromDb()
    train_StateAction = string[0]
    train_label = string[1]
    for i in range(3000):
        file_read_rewardandnextstate = open('D:/brain_computing/QLearingEnvironment/src/main/resources/nextState.json',"r+")
        string = json.load(file_read_rewardandnextstate)
        file_read_rewardandnextstate.close()
        if i != 0 and i % 300 == 0:
            string = GetDataFromDb()
            train_StateAction = string[0]
            train_label = string[1]
        start = time.clock()
        print("这是第%d次训练:"%(i + 1))
        state = [np.random.randint(0,9),np.random.randint(0,2),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,2),np.random.randint(0,9),np.random.randint(0,9)]
        decide_action = DecideDevelopOrExplore()
        if decide_action == 1:
            print("\t这一次的决定是随机采取一个动作！")
            action = np.random.randint(0,9)
        else:
            print("\t这一次的决定是采取一个使当前state的期望最大的action！")
            maxQ_value = 0
            action = 1
            for j in range(9):
                targetQ = Targetnet(state,j)
                if targetQ > maxQ_value:
                    maxQ_value = targetQ
                    action = j
        reward_nextstate = EnvTest(state,action,np.random.randint(0,9))
        print("环境探测成功！")
        reward = reward_nextstate[0]
        next_state = reward_nextstate[1]
        TupleIntoExperience(state,action,reward,next_state)
        Mainnet(train_StateAction,train_label)
        print("this round of train use time:%d"%(time.clock() - start))
        
        





#接下来是测试以及battle阶段

def testtheendofdqn():
    teststate = [1, 0, 4, 4, 6, 0, 8, 6]
    action = 4.000000
    reward = 0.000000
    nextstate = [1, 0, 4, 4, 6, 0, 8, 6]
    
    print(Targetnet(teststate,action))
    print(MakeLabel(teststate,action,reward,nextstate))
    
    string = GetDataFromDb()
    teststate_action = string[0]
    label = string[1]

    for i in range(30):
        print("state and action:")
        print(teststate_action[i])
        print('label and targetQ:')
        print(label[i])
        for j in range(8):
            teststate[j] = teststate_action[i][j]
        action = teststate_action[i][8]
        print(Targetnet(teststate,action))
    
def ChooseAction(state):
    maxQ_value = 0
    action = 0
    for i in range(9):
        targetQ_value = Targetnet(state,i)
        if targetQ_value > maxQ_value:
            action = i
    return action



import traceback    
def BattleOnDqn():
    x_time = input("Please input a number that you have never used!")
    init_state = [8,1,0,0,8,1,0,8]
    action = [' ' for i in range(9)]
    action[0] = "NULL"
    action[1] = "goForward"
    action[2] = "goBack"
    action[3] = "holdHead"
    action[4] = "holdBody"
    action[5] = "holdLeg"
    action[6] = "attackHead"
    action[7] = "attackBody"
    action[8] = "attackLeg"
    '''
    第一次尝试，总共两个玩家
    cbz和zkx
    InitState = [8,1,0,0,8,1,0,8]
    for cbz input of targetnet is[8,1,0,0,8,1,0,8,](正序)
    for zkx input of targetnet is[8,1,0,8,8,1,0,0,](倒序)
    '''
    state = init_state
    round_num = 1
    try:
        file = open("Battle/BattleOnDqn/BattleOnDqn" + x_time + ".txt","w")
        while state[0] != 0 and state[4] != 0:
            #choose action for cbz
            action_cbz = ChooseAction(state)
            #choose action for zkx
            state_zkx = [0 for i in range(8)]
            for i in range(8):
                tmp = state[i]
                i = (i + 4) % 8
                state_zkx[i] = tmp
            action_zkx = ChooseAction(state_zkx)
            string = EnvTest(state,action_cbz,action_zkx)
            nextstate = string[1]
            file.write('Round%d:'%(round_num) + '\n')
            round_num = round_num + 1
            state_string_of_cbz = 'cbz state:\nblood:%d\njustaction:%s\nplace:%d'%(state[0],action[state[2]],state[3])
            state_string_of_zkx = 'zkx state:\nblood:%d\njustaction:%s\nplace:%d'%(state[4],action[state[6]],state[7])
            file.write(state_string_of_cbz + '\n')
            file.write(state_string_of_zkx + '\n')
            file.write('\n')
            state = nextstate
        file.close()
    except BaseException:
        file.close()
    file = open("Battle/BattleOnDqn/BattleOnDqn" + x_time + ".txt","a")
    file.write('Round%d:'%(round_num) + '\n')
    round_num = round_num + 1
    state_string_of_cbz = 'cbz state:\nblood:%d\njustaction:%s\nplace:%d'%(state[0],action[state[2]],state[3])
    state_string_of_zkx = 'zkx state:\nblood:%d\njustaction:%s\nplace:%d'%(state[4],action[state[6]],state[7])
    file.write(state_string_of_cbz + '\n')
    file.write(state_string_of_zkx + '\n')
    file.close()
    if state[0] == 0 and state[4] != 0:
        print("zkx winned!")
    if state[0] !=0 and state[4] == 0:
        print("cbz winned!")
    if state[0] == 0 and state[4] == 0:
        print("They both die!")
        
def BattleOnEnv():
    x_time = input("Please input a number that you have never used!")
    init_state = [8,1,0,0,8,1,0,8]
    action = [' ' for i in range(9)]
    action[0] = "NULL"
    action[1] = "goForward"
    action[2] = "goBack"
    action[3] = "holdHead"
    action[4] = "holdBody"
    action[5] = "holdLeg"
    action[6] = "attackHead"
    action[7] = "attackBody"
    action[8] = "attackLeg"
    '''
    第一次尝试，总共两个玩家
    cbz和zkx
    InitState = [8,1,0,0,8,1,0,8]
    for cbz input of targetnet is[8,1,0,0,8,1,0,8,](正序)
    for zkx input of targetnet is[8,1,0,8,8,1,0,0,](倒序)
    '''
    state = init_state
    round_num = 1
    try:
        file = open("Battle/BattleOnEnv/BattleOnEnv" + x_time + ".txt","w")
        while state[0] != 0 and state[4] != 0:
            #choose action for cbz
            action_cbz = 0
            action_reward = -100
            for i in range(9):
                string = EnvTest(state,i,np.random.randint(0,9))
                if string[0] > action_reward:
                    action_reward = string[0]
                    action_cbz = i
            #choose action for zkx
            state_zkx = [0 for i in range(8)]
            for i in range(8):
                tmp = state[i]
                i = (i + 4) % 8
                state_zkx[i] = tmp
            action_zkx = 0
            action_reward = -100
            for i in range(9):
                string = EnvTest(state_zkx,i,np.random.randint(0,9))
                if string[0] > action_reward:
                    action_reward = string[0]
                    action_zkx = i
            string = EnvTest(state,action_cbz,action_zkx)
            nextstate = string[1]
            file.write('Round%d:'%(round_num) + '\n')
            round_num = round_num + 1
            state_string_of_cbz = 'cbz state:\nblood:%d\njustaction:%s\nplace:%d'%(state[0],action[state[2]],state[3])
            state_string_of_zkx = 'zkx state:\nblood:%d\njustaction:%s\nplace:%d'%(state[4],action[state[6]],state[7])
            file.write(state_string_of_cbz + '\n')
            file.write(state_string_of_zkx + '\n')
            file.write('\n')
            state = nextstate
        file.close()
    except BaseException:
        file.close()
    file = open("Battle/BattleOnEnv/BattleOnEnv" + x_time + ".txt","a")
    file.write('Round%d:'%(round_num) + '\n')
    state_string_of_cbz = 'cbz state:\nblood:%d\njustaction:%s\nplace:%d'%(state[0],action[state[2]],state[3])
    state_string_of_zkx = 'zkx state:\nblood:%d\njustaction:%s\nplace:%d'%(state[4],action[state[6]],state[7])
    file.write(state_string_of_cbz + '\n')
    file.write(state_string_of_zkx + '\n')
    file.close()
    if state[0] == 0 and state[4] != 0:
        print("zkx winned!")
    if state[0] !=0 and state[4] == 0:
        print("cbz winned!")
    if state[0] == 0 and state[4] == 0:
        print("They both die!")