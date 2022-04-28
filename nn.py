import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
from edward.models import Normal, OneHotCategorical
import statsmodels.api as sm
import plot
from util import config


vars = config['experiments']['variables']

activation1 = tf.tanh
#activation1 = tf.nn.relu
activation2 = tf.nn.softmax
learning_rate =0.1
n_hidden=15
n_iter=1000
batch_size=8


# def get_vars():
#     fixed_effect = ['Trip_distance', 'Trip_time', 'Household_employeed',
#        'Household_children', 'Household_bike', 'Household_car',
#        'Household_licence', 'Individual_age', 'Individual_income',
#        'Population_density', 'Trip_purpose_1', 'Trip_purpose_2',
#        'Trip_purpose_3', 'Trip_purpose_4', 'Trip_purpose_5', 'Trip_purpose_6',
#         'Household_settlement_1',
#       # 'Individual_employment_1',
#        'Individual_education_1',
#        'Individual_gender_1'
#                 ]
#     return fixed_effect


def train_test_index(data):
    train_index = data.sample(frac=0.8,random_state=42).index
    test_index = data.drop(train_index).index
    return train_index,test_index

def REBNN_MODEL(data,name,n_hidden=n_hidden,learning_rate=learning_rate,test=False):
    if test==True: print(f'Start training RE-BNN {name} n:{n_hidden} lr:{learning_rate}.')
    if test==False: print(f'Start training RE-BNN {name}.')
    random_effect = data['Household_region'].astype('category').cat.codes
    train_index,test_index=train_test_index(data)

    fixed_effect = vars
    fixed_effect.remove('Population_density')

    X = data[fixed_effect].values
    y = pd.get_dummies(data['Mode'],prefix='Mode').values
    y=y.astype(int)
    X_train = X[train_index]
    y_train = y[train_index]
    random_effect_train=random_effect[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    random_effect_test=random_effect[test_index]

    ##########################  Parameter  #########################################
    dData=X_train.shape[1]
    K = y_train.shape[1]
    n_random=len(set(random_effect_train))

    ########################## Model ##########################################
    def RE_BNN(X,b_r_ph):
        a = activation1(tf.matmul(X, W_0) + b_0)
        a = tf.matmul(a, W_1) + b_1+tf.gather(b_r,b_r_ph)
        return a

    with tf.name_scope("model"):
        X = tf.placeholder(tf.float32, [None,dData], name="X")
        Y_ = tf.placeholder(tf.int64, [None, K], name="Y_input")
        b_r_ph=tf.placeholder(tf.int32,[None])
        W_0 = Normal(loc=tf.zeros([dData, n_hidden], dtype=tf.float32), scale=tf.ones([dData, n_hidden], dtype=tf.float32), name="W_0")
        W_1 = Normal(loc=tf.zeros([n_hidden, K], dtype=tf.float32), scale=tf.ones([n_hidden, K], dtype=tf.float32), name="W_1")
        b_0 = Normal(loc=tf.zeros(n_hidden, dtype=tf.float32), scale=tf.ones(n_hidden, dtype=tf.float32), name="b_0")
        b_1 = Normal(loc=tf.zeros(K, dtype=tf.float32), scale=tf.ones(K, dtype=tf.float32), name="b_1")
        b_random=tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
        b_r=Normal(loc=tf.zeros([n_random,K], dtype=tf.float32), scale=b_random*tf.ones([n_random,K], name="b_r"))
        Y = OneHotCategorical(probs=activation2(RE_BNN(X,b_r_ph)), name="out")

    ########################################## Posterior  #########################################
    if test==False:
        with tf.variable_scope("posterior",reuse=tf.AUTO_REUSE):
            with tf.variable_scope("qW_0"):
                loc = tf.get_variable("loc", [dData, n_hidden])
                scale = tf.nn.softplus(tf.get_variable("scale", [dData, n_hidden]))
                qW_0 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qW_1"):
                loc = tf.get_variable("loc", [n_hidden, K])
                scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden, K]))
                qW_1 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_0"):
                loc = tf.get_variable("loc", [n_hidden])
                scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden]))
                qb_0 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_1"):
                loc = tf.get_variable("loc", [K])
                scale = tf.nn.softplus(tf.get_variable("scale", [K]))
                qb_1 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_r"):
                loc = tf.Variable(tf.random_normal([n_random,K]))
                scale = tf.nn.softplus(tf.Variable(tf.random_normal([n_random,K])))
                qb_r = Normal(loc=loc, scale=scale)
    if test==True:
        with tf.variable_scope(f"posterior{name}"):
            with tf.variable_scope("qW_0"):
                loc = tf.get_variable("loc", [dData, n_hidden])
                scale = tf.nn.softplus(tf.get_variable("scale", [dData, n_hidden]))
                qW_0 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qW_1"):
                loc = tf.get_variable("loc", [n_hidden, K])
                scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden, K]))
                qW_1 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_0"):
                loc = tf.get_variable("loc", [n_hidden])
                scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden]))
                qb_0 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_1"):
                loc = tf.get_variable("loc", [K])
                scale = tf.nn.softplus(tf.get_variable("scale", [K]))
                qb_1 = Normal(loc=loc, scale=scale)
            with tf.variable_scope("qb_r"):
                loc = tf.Variable(tf.random_normal([n_random,K]))
                scale = tf.nn.softplus(tf.Variable(tf.random_normal([n_random,K])))
                qb_r = Normal(loc=loc, scale=scale)

    ########################################## Inference  #########################################
    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1,b_r: qb_r}, data={X:X_train,b_r_ph:random_effect_train,Y:y_train})

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    inference.initialize(n_iter=n_iter,optimizer = optimizer)

    ######################################### Evaluation #########################################
    def RE_BNN_para(X,b_r_ph):
        a = activation1(tf.matmul(X, qw_0_value) + qb_0_value)
        a = tf.matmul(a, qw_1_value) + qb_1_value+tf.gather(qb_r_value,b_r_ph)
        return a

    ######################################### Run  ########################################
    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()
    for _ in range(n_iter):
        inference.update()

    result_para = sess.run([qW_0.mean(), qW_1.mean(), qb_0.mean(), qb_1.mean(), qb_r.mean()])

    qw_0_value = result_para[0]
    qw_1_value = result_para[1]
    qb_0_value = result_para[2]
    qb_1_value = result_para[3]
    qb_r_value = result_para[4]

    if test==False:
        correct = tf.equal(tf.argmax(RE_BNN_para(X,b_r_ph), 1), tf.argmax(Y_, 1))
        y_prob=activation2(RE_BNN_para(X,b_r_ph))

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result_pre= sess.run([accuracy,y_prob], feed_dict={X: X_test, b_r_ph:random_effect_test,Y_ : y_test})

        Test_accuracy=result_pre[0]
        probabilities=result_pre[1]

        return Test_accuracy,probabilities, y_test,result_para

    if test==True:
        correct = tf.equal(tf.argmax(RE_BNN_para(X, b_r_ph), 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        Test_accuracy = sess.run(accuracy, feed_dict={X: X_test, b_r_ph: random_effect_test, Y_: y_test})

        return Test_accuracy


def BNN_MODEL(data, name):
    print(f'Start training BNN{name}.')
    train_index,test_index=train_test_index(data)
    fixed_effect = vars
    X = data[fixed_effect].values
    y = pd.get_dummies(data['Mode'], prefix='Mode').values
    y = y.astype(int)

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    ##########################  Parameter  #########################################
    dData = X_train.shape[1]
    K = y_train.shape[1]

    ########################## Model ##########################################
    def BNN(X):
        a = activation1(tf.matmul(X, W_0) + b_0)
        a = tf.matmul(a, W_1) + b_1
        a = activation2(a)
        return a

    with tf.name_scope("pmodel"):
        X = tf.placeholder(tf.float32, [None, dData], name="X")
        W_0 = Normal(loc=tf.zeros([dData, n_hidden], dtype=tf.float32),
                     scale=tf.ones([dData, n_hidden], dtype=tf.float32), name="W_0")
        W_1 = Normal(loc=tf.zeros([n_hidden, K], dtype=tf.float32), scale=tf.ones([n_hidden, K], dtype=tf.float32),
                     name="W_1")
        b_0 = Normal(loc=tf.zeros(n_hidden, dtype=tf.float32), scale=tf.ones(n_hidden, dtype=tf.float32), name="b_0")
        b_1 = Normal(loc=tf.zeros(K, dtype=tf.float32), scale=tf.ones(K, dtype=tf.float32), name="b_1")
        Y = OneHotCategorical(probs=BNN(X), name="out")

        Y_ = tf.placeholder(tf.int64, [None, K], name="Y_input")

    ########################################## Posterior  #########################################
    with tf.variable_scope("posterior",reuse=tf.AUTO_REUSE):
        with tf.variable_scope("qW_0"):
            loc = tf.get_variable("loc", [dData, n_hidden])
            scale = tf.nn.softplus(tf.get_variable("scale", [dData, n_hidden]))
            qW_0 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qW_1"):
            loc = tf.get_variable("loc", [n_hidden, K])
            scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden, K]))
            qW_1 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qb_0"):
            loc = tf.get_variable("loc", [n_hidden])
            scale = tf.nn.softplus(tf.get_variable("scale", [n_hidden]))
            qb_0 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qb_1"):
            loc = tf.get_variable("loc", [K])
            scale = tf.nn.softplus(tf.get_variable("scale", [K]))
            qb_1 = Normal(loc=loc, scale=scale)

    ########################################## Inference  #########################################
    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1}, data={X: X_train,  Y: y_train})
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    inference.initialize(n_iter=n_iter,optimizer = optimizer)

    ########################################### Evaluation ####################################
    def BNN_para(X):
        a = activation1(tf.matmul(X, qw_0_value) + qb_0_value)
        a = tf.matmul(a, qw_1_value) + qb_1_value
        return a

    ######################################### Run  ########################################
    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    for _ in range(inference.n_iter):
        inference.update()

    # Evaluation
    result_para= sess.run([qW_0.mean(),qW_1.mean(),qb_0.mean(),qb_1.mean()])

    qw_0_value=result_para[0]
    qw_1_value = result_para[1]
    qb_0_value = result_para[2]
    qb_1_value = result_para[3]

    correct = tf.equal(tf.argmax(BNN_para(X), 1), tf.argmax(Y_, 1))
    y_prob=activation2(BNN_para(X))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    result_pre= sess.run([accuracy,y_prob], feed_dict={X: X_test,Y_ : y_test})

    Test_accuracy=result_pre[0]
    probabilities=result_pre[1]

    return Test_accuracy,probabilities, result_para


def DNN_MODEL(data,name,n_hidden=10,learning_rate=0.01,batch_size=100):
    print(f'Start training DNN for {name}')
    train_index,test_index=train_test_index(data)
    X = data[fixed_effect].values
    y = pd.get_dummies(data['Mode'], prefix='Mode').values
    y = y.astype(int)

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    ##########################  Parameter  #########################################
    nData = X_train.shape[0]
    dData = X_train.shape[1]
    K = y_train.shape[1]

    w_0=tf.Variable(tf.truncated_normal([dData,n_hidden],stddev=0.1))
    w_1=tf.Variable(tf.truncated_normal([n_hidden,K],stddev=0.1))
    b_0=tf.Variable(tf.constant(0.1,shape=[n_hidden]))
    b_1=tf.Variable(tf.constant(0.1,shape=[K]))

    def neural_network(X):
        a = activation1(tf.matmul(X, w_0) + b_0)
        a = tf.matmul(a, w_1) + b_1
        return a

    X = tf.placeholder(tf.float32, [None, dData], name="X_input")
    Y_=tf.placeholder(tf.int64, [None,K], name="Y_input")

    Y = neural_network(X)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Y, labels =tf.argmax(Y_,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        INIT_OP = tf.global_variables_initializer()
        sess.run(INIT_OP)
        for i in range(n_iter):
            start=(i*batch_size)%nData
            end=min(start+batch_size,nData )
            sess.run(train_step,feed_dict={X:X_train[start:end],Y_:y_train[start:end]})
        #evaluation
        result_para = sess.run([w_0, w_1,b_0,b_1])
        y_prob = activation2(neural_network(X))
        correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        result_pre= sess.run([accuracy,y_prob], feed_dict={X: X_test,  Y_: y_test})

    Test_accuracy=result_pre[0]
    probabilities=result_pre[1]

    return Test_accuracy,probabilities, result_para





def REDNN_MODEL(data,name,n_hidden=10,learning_rate=0.01,batch_size=100):
    print(f'Start training DNN for {name}')
    random_effect = data['Household_region'].astype('category').cat.codes
    train_index,test_index=train_test_index(data)
    fixed_effect = vars
    fixed_effect.remove('Population_density')
    X = data[fixed_effect].values
    y = pd.get_dummies(data['Mode'], prefix='Mode').values
    y = y.astype(int)

    X_train = X[train_index]
    y_train = y[train_index]
    random_effect_train = random_effect[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    random_effect_test=random_effect[test_index]

    ##########################  Parameter  #########################################
    nData = X_train.shape[0]
    dData = X_train.shape[1]
    K = y_train.shape[1]
    n_random = len(set(random_effect_train))

    w_0=tf.Variable(tf.truncated_normal([dData,n_hidden],stddev=0.1))
    w_1=tf.Variable(tf.truncated_normal([n_hidden,K],stddev=0.1))
    b_0=tf.Variable(tf.constant(0.1,shape=[n_hidden]))
    b_1=tf.Variable(tf.constant(0.1,shape=[K]))
    b_random = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
    b_r = Normal(loc=tf.zeros([n_random, K], dtype=tf.float32), scale=b_random * tf.ones([n_random, K], name="b_r"))

    def neural_network(X,b_r_ph):
        a = activation1(tf.matmul(X, w_0) + b_0)
        a = tf.matmul(a, w_1) + b_1+tf.gather(b_r,b_r_ph)
        return a

    X = tf.placeholder(tf.float32, [None, dData], name="X_input")
    Y_=tf.placeholder(tf.int64, [None,K], name="Y_input")
    b_r_ph = tf.placeholder(tf.int32, [None])
    Y = neural_network(X,b_r_ph)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Y, labels =tf.argmax(Y_,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        INIT_OP = tf.global_variables_initializer()
        sess.run(INIT_OP)
        for i in range(n_iter):
            start=(i*batch_size)%nData
            end=min(start+batch_size,nData )
            sess.run(train_step,feed_dict={X:X_train[start:end],b_r_ph: random_effect_train[start:end],Y_:y_train[start:end]})
        #evaluation
        result_para = sess.run([w_0, w_1,b_0,b_1,b_r])
        y_prob = activation2(neural_network(X,b_r_ph))
        correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        result_pre= sess.run([accuracy,y_prob], feed_dict={X: X_test, b_r_ph: random_effect_test, Y_: y_test})

    Test_accuracy=result_pre[0]
    probabilities=result_pre[1]

    return Test_accuracy,probabilities, result_para

def get_vars_MNL(data,var=1,print_detail=True):
    '''
    This function is used to print the correlation coefficients between variables and each alternatives
    and select variables for each alternative-specific function.
    :param data: A dataframe containing variables and alternatives (labels)
    :param var: The index of a variable that each function specification in MNL model will contains.
     It is necessary for the choice probability analysis latter.
     Default is 'Trip_time'
    :return: A list containing selected variables for each alternative-specific function.
    '''

    data_1 = pd.merge(data, data.Mode.replace([2, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_2 = pd.merge(data, data.Mode.replace([1, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_3 = pd.merge(data, data.Mode.replace([1, 2, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_4 = pd.merge(data, data.Mode.replace([1, 2, 3, 5], [0, 0, 0, 0]), on=data.index)
    data_5 = pd.merge(data, data.Mode.replace([1, 2, 3, 4], [0, 0, 0, 0]), on=data.index)
    data_mode = [data_1, data_2, data_3, data_4, data_5]

    mode_name = plot.get_modes()

    vars_MNL=[]
    corr = np.zeros((len(mode_name),len(vars)))
    if print_detail==True:
        for i in range(len(data_mode)):
            data_corr=data_mode[i]
            corr[i]=data_corr.drop(columns=['Mode_x','key_0']).corr()['Mode_y'].reindex(vars,axis = 'rows').values
            corr_col = data_corr.drop(columns=['Mode_x','key_0']).corr()['Mode_y'].abs().sort_values(ascending=False).head(10)
            print(f'Columns correlated with the alternatives {mode_name[i]}:')
            print(corr_col)
            data_1_col=[]
            for j in range(1,10):
                if corr_col[corr_col.index[j]]>0.1:data_1_col.append(corr_col.index[j])
            if len(data_1_col)<2:
                data_1_col=list(corr_col.index[1:3])
            if vars[var] not in data_1_col:
                data_1_col.append(vars[var])
            print(f'Columns selected in the utility function of {mode_name[i]}:')
            print(data_1_col)
            print(' ')
            vars_MNL.append(data_1_col)
    else:
        for i in range(len(data_mode)):
            data_corr = data_mode[i]
            corr_col = data_corr.drop(columns=['Mode_x']).corr()['Mode_y'].abs().sort_values(ascending=False).head(10)
            data_1_col = []
            for j in range(1, 10):
                if corr_col[corr_col.index[j]] > 0.1: data_1_col.append(corr_col.index[j])
            if len(data_1_col) < 2:
                data_1_col = list(corr_col.index[1:3])
            if vars[var] not in data_1_col:
                data_1_col.append(vars[var])
            vars_MNL.append(data_1_col)

    return vars_MNL,corr.T


def MNL_MODEL(data,vars_MNL):
    data_1 = pd.merge(data, data.Mode.replace([2, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_2 = pd.merge(data, data.Mode.replace([1, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_3 = pd.merge(data, data.Mode.replace([1, 2, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_4 = pd.merge(data, data.Mode.replace([1, 2, 3, 5], [0, 0, 0, 0]), on=data.index)
    data_5 = pd.merge(data, data.Mode.replace([1, 2, 3, 4], [0, 0, 0, 0]), on=data.index)
    data_mode = [data_1, data_2, data_3, data_4, data_5]

    train_index, test_index = train_test_index(data)

    def utility_mnl_one_mode(data_1, mode):
        train_index, test_index = train_test_index(data_1)
        data_train = data_1.loc[train_index]
        data_pre = data_1.loc[test_index]

        data_1_col=vars_MNL[mode]

        model_1 = sm.OLS(exog=data_train[data_1_col], endog=data_train.Mode_y).fit()
        utility_pred_1 = model_1.predict(data_pre[data_1_col])
        return utility_pred_1

    utility_all_mode = np.zeros((len(data_mode), len(utility_mnl_one_mode(data_1, 1))))
    for i in range(len(data_mode)):
        utility_all_mode[i] = utility_mnl_one_mode(data_mode[i], i)

    pre = np.exp(utility_all_mode) / np.sum(np.exp(utility_all_mode), axis=0)

    probability=pre.T

    pre_onehot = np.copy(probability)
    for j in range(len(probability)):
        pre_list = list(probability[j])
        ind = pre_list.index(max(pre_list))
        for i in range(len(pre_list)):
            if i == ind:
                pre_onehot[j][i] = 1
            else:
                pre_onehot[j][i] = 0

    # cheack the accuracy
    def cat_accuracy(true, pre):
        right = 0
        for i in range(len(true)):
            if list(true[i]).index(1) == list(pre[i]).index(1): right += 1
        return right / len(true)

    y = data['Mode'].values
    y_test_df = pd.DataFrame(y[test_index],columns=['choice'])
    y_test=pd.get_dummies(y_test_df['choice'], prefix='choice').values

    return cat_accuracy(y_test, pre_onehot),probability


