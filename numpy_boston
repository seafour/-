# 使用numpy实现Boston房价预测
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

#建立模型 
class boston_price_model:
    def __init__(self):
        self.n_features = 13
        self.n_hidden = 10
        self.w1 = np.random.randn(self.n_features, self.n_hidden)
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = np.random.randn(self.n_hidden, 1)
        self.b2 = np.zeros(1)

    def update_model(self,w1,w2,b1,b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    
    def predict(self,X):
        li = Linear(X, self.w1, self.b1)
        s1 = Relu(li)
        y_pred = Linear(s1, self.w2, self.b2)
        return y_pred

# 数据加载
data = load_boston()
X_ = data['data']
Y_ = data['target']
#print(X_)
#print(len(X_)) # 506
Y_ = Y_.reshape(Y_.shape[0],1)

    

# 数据规范化
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
#Y_ = (Y_ - np.mean(Y_, axis=0)) / np.std(Y_, axis=0)

#模型初始化
n_features = X_.shape[1]
n_hidden = 10
W1 = np.random.randn(n_features, n_hidden)
B1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, 1)
B2 = np.zeros(1)

# relu函数
def Relu(x):
    result = np.where(x<0,0,x)
    return result

# 设置学习率
learning_rate = 1e-4

def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))

def Linear(X, W1, b1):
    result = X.dot(W1) + b1
    return result

# 每个批次进行100次迭代 
trytimes = 5000
# 将数据分成4*100的训练集和1*106的预测集
batch = 4
batch_num = 100 


sum1 = W1.copy()
sum1 = np.zeros(sum1.shape)
sum2 = W2.copy()
sum2 = np.zeros(sum2.shape)

for i in range(batch):
    #每一个批次的权重和bias都需要重新计算 
    w1 = np.random.randn(n_features, n_hidden)
    b1 = np.zeros(n_hidden)
    w2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros(1)
    for t in range(trytimes):
        # 计算前向传播
        x = X_[i*batch_num:(i+1)*batch_num,:]
        y = Y_[i*batch_num:(i+1)*batch_num]
        
        l1 = Linear(x, w1, b1)
        s1 = Relu(l1)
        #print(x,x.shape)
        
        y_pred = Linear(s1, w2, b2)
        
       # print(y,y.shape)
        #print(s1,s1.shape)
       # print(y_pred,y_pred.shape)
        #计算损失函数
        loss = MSE_loss(y, y_pred)
        print(t, loss)
        # 反向传播， 基于loss计算w1 和 w2的梯度
        grad_y_pred = 2.0* (y_pred - y)
       # print(grad_y_pred.shape)
        grad_w2 = s1.T.dot(grad_y_pred)
        
        grad_temp_relu = grad_y_pred.dot(w2.T)
        grad_temp = grad_temp_relu.copy()
        grad_temp_relu[l1<0] = 0
        grad_w1 = x.T.dot(grad_temp)
        
        #更新权重
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    
    # 计算每个批次的 w1 w2的平均值 
    sum1 += w1
    W1 = sum1 / (i + 1)
    sum2 += w2
    W2 = sum2 / (i + 1)

print('w1={} \n w2={}'.format(W1, W2))

# 保存模型参数
mymodel = boston_price_model()
mymodel.update_model(W1, W2, B1, B2)

# 开始预测
y_pred = mymodel.predict(X_[400:506,:])
loss = MSE_loss(Y_[400:506], y_pred)

comp_arr = np.array([y_pred.reshape(106), Y_[400:506].reshape(106)])
comp_arr.reshape(2,106)
print(comp_arr)

print(loss)

# =============================================================================
# for t in range(5000):
#     # 前向传播，计算预测值y
#     l1 = Linear(X_, w1, b1)
#     s1 = Relu(l1)
#     y_pred = Linear(s1, w2, b2)
#     
#     # 计算损失函数
#     loss = MSE_loss(y, y_pred)
#     print(t, loss)
# 
#     # 反向传播，基于loss 计算w1和w2的梯度
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = s1.T.dot(grad_y_pred) #(10, 506) * (506, 506)
# 
#     grad_temp_relu = grad_y_pred.dot(w2.T) #(506, 506),  (1, 10)
#     grad_temp = grad_temp_relu.copy()
#     grad_temp_relu[l1<0] = 0
#     grad_w1 = X_.T.dot(grad_temp)
# 
#     # 更新权重
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2
# print('w1={} \n w2={}'.format(w1, w2))
# =============================================================================

