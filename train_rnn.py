import tensorflow as tf
import numpy as np
from layers import Dense, NALU, GLU, GLUv2, SRUCell, UGRnnCell, SimpleAttention, SelfAttention, AttentionLayer
from rnn_net import SimpleRNN

#{"BULLSHIT":2,"OTHER":1,"DRONE":0}
# clouds video_id - 0

BATCH_SIZE = 64
DATA_PATH = "./train_data/data_seq_test_10.npz"
SEQ_LEN = 10
LEARNING_RATE = 0.001
MAX_STEPS = 20000
TRAIN_TEST_SPLIT = 0.33
LOG_ITER = 100
CLASS_WEIGTH = {0:1, 1:5}
ZR_ONLY = True
gen = np.random.RandomState(42)

data = np.load(DATA_PATH)
X = data["X"]
Y_ = data["Y"]
ZR = data["ZR"]
VID = data["VID"]
Y = np.ones_like(Y_,dtype=np.float32)
Y[Y_!="DRONE"] = 0

def fill_mat(mat,y,y_true,labels):
	for i,l in enumerate(["DRONE","OTHER","BULLSHIT"]):
		mask1 = labels == l
		acc = y[mask1] == y_true[mask1]
		mat[i] = np.mean(acc.astype(np.float32))

def print_mat(mat):
	for i,l in enumerate(["DRONE","OTHER","BULLSHIT"]):
		print("{:<10} | {}".format(l,mat[i]))
mat_ideal = np.zeros((3,1),dtype=np.float32)
mat_model = np.zeros((3,1),dtype=np.float32)

mat_ideal_clouds_only = np.zeros((3,1),dtype=np.float32)
mat_model_clouds_only = np.zeros((3,1),dtype=np.float32)

clouds_only_x = X[VID==0]
clouds_only_zr = ZR[VID==0]
clouds_only_y = Y[VID==0]
clouds_only_labels = Y_[VID==0]

train_x = []
train_y = []
train_zr = []
train_labels = []
test_x = []
test_y = []
test_zr = []
test_labels = []
clound_only_test, clound_only_train = 0,0
for t in [0,1]:
	i = Y==t
	idx = gen.rand(np.sum(i)) > TRAIN_TEST_SPLIT
	clound_only_train += np.sum((VID==0)[i][idx])
	clound_only_test += np.sum((VID==0)[i][~idx])
	train_x.append(X[i][idx])
	test_x.append(X[i][~idx])
	train_y.append(Y[i][idx])
	test_y.append(Y[i][~idx])
	train_zr.append(ZR[i][idx])
	test_zr.append(ZR[i][~idx])
	train_labels.append(Y_[i][idx])
	test_labels.append(Y_[i][~idx])
train_x = np.concatenate(train_x,0)
train_y = np.concatenate(train_y,0)
test_x = np.concatenate(test_x,0)
test_y = np.concatenate(test_y,0)
train_zr = np.concatenate(train_zr,0)
test_zr = np.concatenate(test_zr,0)
train_labels = np.concatenate(train_labels,0)
test_labels = np.concatenate(test_labels,0)
train_w = np.zeros((train_y.shape[0],1))
test_w = np.zeros((test_y.shape[0],1))
for k in CLASS_WEIGTH:
	train_w[train_y==k] = CLASS_WEIGTH[k]
	test_w[test_y==k] = CLASS_WEIGTH[k]

fill_mat(mat_ideal,test_zr,test_y,test_labels)
fill_mat(mat_ideal_clouds_only,clouds_only_zr,clouds_only_y,clouds_only_labels)

if ZR_ONLY:
	train_x = train_x[train_zr==1]
	train_y = train_y[train_zr==1]
	train_labels = train_labels[train_zr==1]
	test_labels = test_labels[test_zr==1]
	test_x = test_x[test_zr==1]
	test_y = test_y[test_zr==1]

	clouds_only_x = clouds_only_x[clouds_only_zr==1]
	clouds_only_y = clouds_only_y[clouds_only_zr==1]
	clouds_only_labels = clouds_only_labels[clouds_only_zr==1]

#rnn_cell = tf.contrib.rnn.LSTMCell(28)
#rnn_cell = SRUCell(28,tf.nn.selu)
rnn_cell = UGRnnCell(28,tf.nn.selu)
overstructure = [Dense(20,tf.nn.selu),Dense(14,tf.nn.selu)] #SimpleAttention(28,tf.nn.selu)
#overstructure = [AttentionLayer(SelfAttention(5,40,50,30),20,tf.nn.selu),AttentionLayer(SelfAttention(5,30,40,20),14,tf.nn.selu)]

model = SimpleRNN(rnn_cell,overstructure,seq_len=SEQ_LEN,feature_len=train_x.shape[-1],learning_rate=LEARNING_RATE)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	model.initialize(sess)

	for i in range(MAX_STEPS):
		idx = gen.randint(0, train_y.shape[0], BATCH_SIZE)
		batch_x = train_x[idx]
		batch_y = train_y[idx]
		batch_w = train_w[idx]

		model.train_step(batch_x,batch_y,batch_w)
		step = model.get_global_step()
		predicts_train = model.predict(batch_x)

		if step % LOG_ITER == 0:
			idx = gen.randint(0, test_y.shape[0], BATCH_SIZE)
			batch_x = test_x[idx]
			batch_y = test_y[idx]
			batch_w = test_w[idx]

			acc_total, loss, loss_med = model.get_loss(batch_x,batch_y,batch_w)
			predicts = model.predict(batch_x)
			print("#{} | Loss: {:10} | LossMedian: {:10}| Acc: {:10}".format(step,loss,loss_med,acc_total))

	y_model = model.predict(test_x)
	clouds_only_model = model.predict(clouds_only_x)

	model.to_json("net.json")

fill_mat(mat_model,np.round(y_model),test_y,test_labels)
fill_mat(mat_model_clouds_only,np.round(clouds_only_model),clouds_only_y,clouds_only_labels)

print("Ideal")
print_mat(mat_ideal)
print("Model")
print_mat(mat_model)

print("Clouds train: {}, test: {}".format(clound_only_train, clound_only_test))
print("Ideal (office_clouds_2018-09-19__0__)")
print_mat(mat_ideal_clouds_only)
print("Model (office_clouds_2018-09-19__0__)")
print_mat(mat_model_clouds_only)