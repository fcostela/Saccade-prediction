'''
Multi-Layer Recurrent Neural Network (LSTM) 
Use (x,y) to predict the (x,y) 10 ms into the future
Predict from the first point of saccade

training [0, int(saccade.size * split_ratio * split_tv))
validation [int(saccade.size * split_ratio * split_tv), int(saccade.size * split_ratio))
testing [int(saccade.size * split_ratio), saccade.size)
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as spio
import operator
import os.path

#-----------------Parameters-----------------
path = '/media/luolab/DATA/Projects/Saccade/'
path_model = path + 'Save_General/XY_model_2019/' #folder used to save the model
path_log = path + 'Save_General/XY_logs_2019' #folder used to save the logs
path_data = path + 'Data/'  #data folder

input_size = 2          #the number of features
output_size = 2
#learning_rate = 0.0005
learning_rate = 0.005
rnn_unit = 256           #hidden layer units
batch_size = 250
#time_steps = 40         # time step for each input training
#time_steps = 15         # time step for each input training
time_steps = 22         # time step for each input training
number_of_layers = 5   # number of layers for rnn
time_delay = 10         #time delay for prediction
epochs = 25000

pn_before = 16 #the number of points before the saccade onset
pn_add = time_delay + (time_steps-pn_before) - 1 #the number of points need to add before each saccade

def get_data(flag):
    mat = spio.loadmat(path_data+'saccadeTV1Degree_plus16msbefore.mat')
    saccade = mat['saccadeIndex']  # array of Feature_List
    saccade = saccade[0]

    '''
    if flag == 'train':
        rng = range(1, 2)
    elif flag == 'validation':
        rng = range(88,89)
    else:
        rng = range(95, 97)
    '''

    if flag == 'train':
        rng = range(1, 66)
    elif flag == 'validation':
        rng = range(66, 81)
    elif flag == 'test':
        rng = range(81, 101)
    else:
        rng = range(1, 101)

    feature_list = []
    label_list = []
    batch_idx = []
    for idx in range(saccade.size):
        saccade_struct = saccade[idx]
        if saccade_struct['subNumber'][0][0] not in rng:
            continue

        saccade_one = saccade_struct['saccadeNoFilter']
        saccade_one = saccade_one.tolist()
        saccade_one = np.reshape(saccade_one[0] * pn_add, (pn_add,2)).tolist() + saccade_one

        for i in range(0, len(saccade_one) - time_delay - time_steps + 1):
            if len(feature_list) % batch_size == 0:
                batch_idx.append(len(feature_list))
            feature = saccade_one[i:i + time_steps]
            #y = dis[i + time_delay:i + time_delay + time_steps] # time_steps predicting points
            label = saccade_one[i + time_delay + time_steps-1] #one predicting point
            feature_mean = np.mean(feature, axis=0)
            #feature = [f - feature_mean for f in feature]
            #label = [f - feature_mean for f in label]
            feature = feature - feature_mean
            label = label - feature_mean

            feature_list.append(feature)
            label_list.append(label)
    batch_idx.append(len(feature_list))
    return batch_idx, feature_list, label_list

#-----------------weights & biases-----------------
# Define weights
with tf.name_scope('weights'):
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit]), name='in'),
        'out': tf.Variable(tf.random_normal([rnn_unit, output_size]), name='out')
    }
with tf.name_scope('biases'):
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name='in'),
        'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]), name='out')
    }

w_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
b_out = tf.Variable(tf.random_normal([output_size]))

def RNN(X):
    # it seems the input weight and bias are not necessary
    # w_in = weights['in']
    # b_in = biases['in']
    # input = tf.reshape(X, [-1, input_size]) # change tensor to 2D for computation
    # input_rnn = tf.matmul(input, w_in) + b_in
    # input_rnn = tf.reshape(input_rnn, [-1, time_steps, rnn_unit]) # change tensor to  3D for the input of lstm cell

    input_rnn = tf.unstack(X, time_steps, 1)

    with tf.name_scope('RNNLayers'):
        cells = []
        for _ in range(number_of_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit, forget_bias=0.0, state_is_tuple=True)
            # The terminology is unfortunately inconsistent. num_units in TensorFlow is the number of hidden states, i.e. the dimension of ht
            cells.append(cell)
        lsmt_layers = rnn.MultiRNNCell(cells)

        #output_rnn, final_states = rnn.static_rnn(lsmt_layers, input_rnn, dtype=tf.float32)
        # output_rnn, final_states = tf.nn.dynamic_rnn(lsmt_layers, input_rnn, dtype=tf.float32)
        output_rnn, final_states = tf.nn.static_rnn(lsmt_layers, input_rnn, dtype=tf.float32)

        # output = tf.reshape(output_rnn, [-1, time_steps, rnn_unit])  # the input for the output layer
        # output = tf.reshape(output[:,time_steps-1,:], [-1,rnn_unit]) #only take the last step
        # w_out = weights['out']
        # b_out = biases['out']
        # Predict = tf.matmul(output, w_out) + b_out

        Predict = tf.matmul(output_rnn[-1], w_out) + b_out

        return Predict, final_states

    # output = tf.reshape(output_rnn, [-1, rnn_unit]) # the input for the output layer
    # w_out = weights['out']
    # b_out = biases['out']
    # pred = tf.matmul(output[-1], w_out) + b_out
    # return pred, final_states


def train_lstm():
    epochs_current = tf.Variable(0, "epochs_current")

    # load training data
    batch_idx, train_feature, train_label = get_data('train')

    # load evaluation data
    batch_val_idx, val_feature, val_label = get_data('validation')


    # val_label_gt = [] #validation label ground truth
    # for e in val_label:
    #     val_label_gt = np.append(val_label_gt, np.array(e))
    val_label_gt =np.array(val_label)

    # build model
    with tf.name_scope('Model'):
        Feature = tf.placeholder(tf.float32, shape=[None, time_steps, input_size])
        #Y = tf.placeholder(tf.float32, shape=[None, time_steps, output_size])
        Label = tf.placeholder(tf.float32, shape=[None, output_size])
        Predict,_ = RNN(Feature)

    # training loss
    with tf.name_scope('Loss'):
        loss = tf.losses.mean_squared_error(tf.reshape(Predict, [-1, input_size]), tf.reshape(Label, [-1, input_size]))
    tf.summary.scalar('loss', loss) # create a summary to monitor the loss


    with tf.name_scope('Traing'):
        optimizer = tf.contrib.layers.optimize_loss(
              loss=loss,
              global_step=tf.contrib.framework.get_global_step(),
              learning_rate=learning_rate,
              optimizer="SGD")
    saver = tf.train.Saver()

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(path_log)

    with tf.Session() as sess:
        # devices = sess.list_devices()
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, path_model + "model.ckpt")

        summary_writer.add_graph(sess.graph)

        # # local variables
        # val_loss_max = 100  # maximum validation loss

        #for i in range(epochs):
        for i in range(epochs_current.eval() + 1, epochs):
            # training

            for step in range(len(batch_idx)-1):

                # pred_y = sess.run(pred,
                #                      feed_dict={X: train_x[batch_idx[step]:batch_idx[step + 1]]})
                #
                # y = train_y[batch_idx[step]:batch_idx[step + 1]]

                _, error, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={Feature: train_feature[batch_idx[step]:batch_idx[step + 1]],
                                                        Label: train_label[batch_idx[step]:batch_idx[step + 1]]})
                summary_writer.add_summary(summary, step + i*len(batch_idx))
                if step % 50 == 0:
                    print(i, step, error)

            # evaluation
            val_label_pr = []
            for step in range(len(batch_val_idx) - 1):
                predict = sess.run(Predict, feed_dict={Feature: val_feature[batch_val_idx[step]:batch_val_idx[step + 1]]})
                # test_predict.extend(predict)
                val_label_pr = np.append(val_label_pr, predict)

            val_label_gt = np.reshape(val_label_gt, [-1,2])
            val_label_pr = np.reshape(val_label_pr,[-1,2])

            p_acc_1 = np.average(np.sqrt((val_label_gt[:,1] - val_label_pr[:,1])**2 + (val_label_gt[:,0] - val_label_pr[:,0])**2))
            summary = tf.Summary(value=[tf.Summary.Value(tag='p_acc_1', simple_value=p_acc_1)])
            summary_writer.add_summary(summary, global_step=i)

            assign_op = epochs_current.assign(i)
            sess.run(assign_op)

            # save results
            if i % 25 == 0:
                if not os.path.exists(path_model + str(i)):
                    os.makedirs(path_model + str(i))
                saver.save(sess, path_model + str(i) + "/" + "model.ckpt")


def prediction(flag):
    # load evaluation data
    batch_test_idx, test_feature, test_label = get_data(flag)


    # test_label_gt = [] #validation label ground trouth
    # for e in test_label:
    #     test_label_gt = np.append(test_label_gt, np.array(e))

    test_label_gt = np.array(test_label)

    # batch_idx, test_x, test_y = get_test_data()
    # # e is a list, t is the element of e
    # y = []
    # for e in test_y:
    #     y = np.append(y, np.array(e))

    # build model
    Feature = tf.placeholder(tf.float32, shape=[None, time_steps, input_size])
    Predict,_ = RNN(Feature)


    # x = tf.placeholder(tf.float32, shape=[None, time_steps, input_size])
    # pred, _ = RNN(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path_model + "model.ckpt")

        test_label_pr = []
        for step in range(len(batch_test_idx)-1):
            predict = sess.run(Predict, feed_dict={Feature: test_feature[batch_test_idx[step]:batch_test_idx[step + 1]]})
            # test_predict.extend(predict)
            test_label_pr = np.append(test_label_pr, predict)

        test_label_gt = np.reshape(test_label_gt, [-1, 2])
        test_label_pr = np.reshape(test_label_pr, [-1, 2])
        error_point2point = np.sqrt((test_label_gt[:, 1] - test_label_pr[:, 1]) ** 2 + (test_label_gt[:, 0] - test_label_pr[:, 0]) ** 2)
        p_acc = np.average(error_point2point)
        print(p_acc)
        p_acc_median = np.median(error_point2point)
        print(p_acc_median)
        spio.savemat(path_data + 'error_point2point_general_all.mat',
                        mdict={'error_point2point': error_point2point})


#train_lstm()
#prediction('test')
prediction('all')

'''
https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
'''
