from pylab import *
from matplotlib import pyplot as plt
import time
# 1.导入路径
caffe_root = '../'

train_net_path = './net/lenet_P_train.prototxt'
val_net_path = './net/lenet_P_val.prototxt'
solver_config_path = './net/lenet_P_solver.prototxt'



import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import os
os.chdir(caffe_root)

# 2.创建Lenet-P网络
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):

    n = caffe.NetSpec()
    # 数据层
    n.Data, n.Label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    # 卷积层 1
    n.conv1 = L.Convolution(n.Data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # 池化层 1
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # 卷积层 2
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # 池化层 2
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # 卷积层 3
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=80, weight_filler=dict(type='xavier'))
    # 池化层 3
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=1, pool=P.Pooling.MAX)
    # 全连接层
    n.fc1 =   L.InnerProduct(n.pool3, num_output=500, weight_filler=dict(type='xavier'))
    
    # RELU激活层
    n.relu =  L.ReLU(n.fc1, in_place=True)
    # 输出层
    n.score = L.InnerProduct(n.relu, num_output=10, weight_filler=dict(type='xavier'))
    # 损失层
    n.loss =  L.SoftmaxWithLoss(n.score, n.Label)
    
    return n.to_proto()

    
with open(train_net_path, 'w') as f:
    f.write(str(lenet('./data/train_lmdb', 64)))    
with open(val_net_path, 'w') as f:
    f.write(str(lenet('./data/val_lmdb', 100)))


# 3. 定义solver
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# 设置一个种子控制训练随机化
s.random_seed = 0xCAFFE

s.train_net = train_net_path
s.test_net.append(val_net_path)


s.test_interval = 500    # 每迭代 500次 进行测试
s.test_iter.append(100)  #每次测试 100个

#------------------------AdaDelta---------------------

s.base_lr = 1.0
s.lr_policy = 'fixed'
s.momentum = 0.95
s.weight_decay = 0.0005
s.type = 'AdaDelta'
s.delta = 1e-6

# 最大迭代次数
s.max_iter = 10000
s.display = 1000
s.snapshot = 5000
s.snapshot_prefix = 'model/lenet-P/'

s.solver_mode = caffe_pb2.SolverParameter.GPU

with open(solver_config_path, 'w') as f:
    f.write(str(s))


caffe.set_device(0)
caffe.set_mode_gpu()
# 4. 加载 solver 和 创建训练测试net
solver = None
solver = caffe.get_solver(solver_config_path)

# 打印 数据规模和参数规模
print("------------------------------")
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
print()
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])
print("------------------------------")
	
# 设置迭代3次数
niter = 1001
test_interval = int(niter / 10)



# 迭代模型
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

start = time.time()

for it in range(niter):
    solver.step(1)  
    
    train_loss[it] = solver.net.blobs['loss'].data
    if it % test_interval == 0:
        print('迭代', it, '测试...')
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['Label'].data)
        print(correct)
        print(it)
        test_acc[int(it // test_interval)] = correct / 1e4

end = time.time()
print("训练时长:{}".format(end-start))

fig = plt.figure(figsize=(20,8),dpi=80)
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(arange(niter), train_loss, label = 'loss',color = 'r')
ax2 = ax1.twinx()
print(test_acc)
lns2 = ax2.plot(test_interval * arange(len(test_acc)), test_acc, label = 'accuracy', color = 'b')
# plt.xticks(range(0,501,50))
ax1.set_xlabel("iteration")
ax1.set_ylabel("train loss", color = 'r')
ax2.set_ylabel("test accuracy", color = 'b')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=7)
ax2.set_title('LeNet-5 Test Accuracy: {:.2f} (ReLU)'.format(test_acc[-1]))

plt.grid(True, linestyle = "-", color = "g", linewidth = "1")

plt.show()

"""
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('错误率')
ax2.set_ylabel('识别率')
ax2.set_title('网络识别率: {:.2f}'.format(test_acc[-1]))
plot.savefig("0.png")
"""
