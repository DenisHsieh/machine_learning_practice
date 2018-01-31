import tensorflow as tf 
import numpy as np 

# 建置滿足一元二次方程的函數

# 為了使點更細一些，我們建置了300個點，分佈在 -1 到 1 區間，
# 直接採用 np 產生等差級數的方法，
# 並將結果為 300 個點的一維陣列轉為 300x1 的二維陣列
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

# 加入一些雜訊點，使他與 x_data 的維度一致，並且擬合為平均值 0 、方差為 0.05 的正態分佈
noise = np.random.normal(0, 0.05, x_data.shape)

# y = x^2 - 0.5 + 雜訊
y_data = np.square(x_data) - 0.5 + noise

# 輸入神經網路的變數
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 建置網路模型

def add_layer(inputs, in_size, out_size, activation_function=None):
	
	# 建置加權：in_size x out_size 大小的矩陣
	weights = tf.Variable(tf.random_normal([in_size, out_size]))

	# 建置偏置：1 x out_size 的矩陣
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

	# 矩陣相乘
	Wx_plus_b = tf.matmul(inputs, weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	# 獲得輸出資料 
	return outputs

# 建置隱藏層，假設隱藏層有20個神經元
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)

# 建置輸出層，假設輸出層與輸入層一樣，有 1 個神經元
prediction = add_layer(h1, 20, 1, activation_function=None)

# 計算預測值和真實值之間的誤差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) 

train_step = tf.train.GrandientDescentOptimizer(0.1).minimize(loss)