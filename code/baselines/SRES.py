import math,os,random,sys
import numpy as np
import pandas as pd
import warnings
from statistics import mean, median, mode, stdev
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial, stats
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import svm
from sklearn.metrics import confusion_matrix,f1_score,recall_score,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import train_test_split,StratifiedKFold
from scipy.stats import wilcoxon
import tensorflow as tf

def reset_graph(seed = 42):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)
    np.random.seed(seed)

#Phase I: stage 1: similarity score computation
def Similarity_Score_WSR():
    for row in range(len(list1)):
     te=list1[row]
     PATH=os.path.join(r'./content',te)
     Adata=pd.read_csv(PATH)
     Adata=Adata.drop_duplicates(keep='first')
     ta=Adata.values
     ta_x=ta[:,:-1]
     for colm in range(len(list1)):
       tr=list1[colm]
       arry_te=np.zeros((5,len(Adata.columns)-1))
       arry_tr=np.zeros((5,len(Adata.columns)-1))
       if(tr==te):
        sim_score[row][colm]=1
       else:
        PATH=os.path.join(r'./content', tr)
        train=pd.read_csv(PATH)
        train=train.drop_duplicates(keep='first')
        tra=train.values
        tar_x=tra[:,:-1]
        arry_te[0]=np.mean(ta_x,axis=0)
        arry_tr[0]=np.mean(tar_x,axis=0)
        arry_te[1]=stats.mode(ta_x)[0]
        arry_tr[1]=stats.mode(tar_x)[0]
        arry_te[2]=np.median(ta_x,axis=0)
        arry_tr[2]=np.median(tar_x,axis=0)
        arry_te[2]=np.median(ta_x,axis=0)
        arry_tr[2]=np.median(tar_x,axis=0)
        arry_te[3]=np.std(ta_x,axis=0)
        arry_tr[3]=np.std(tar_x,axis=0)
        arry_te[4]=np.max(ta_x,axis=0) - np.min(ta_x,axis=0)
        arry_tr[4]=np.max(tar_x,axis=0) - np.min(tar_x,axis=0)
        sum_v=0
        temp_cnt=0
        for i in range(5):
            difrn=arry_te[i]-arry_tr[i]
            diff=np.round(difrn,3)
            dif = diff[np.nonzero(diff)]
            arz=np.round(abs(difrn),3)
            ar = arz[np.nonzero(arz)]
            n1=np.count_nonzero(arz)
            ars = pd.Series(ar)
            rnk=ars.rank()
            sig=dif/ar
            sum_value=rnk*sig
            sum_pos=sum_value[sum_value > 0].sum()
            sum_neg=sum_value[sum_value < 0].sum()*-1
            n=n1*(n1+1)
            t=min(sum_pos,sum_neg)
            ttl=sum_pos+sum_neg
            rc1=4*abs(t-(ttl/2))
            rc1=rc1/n
            w,p=wilcoxon(difrn)
            if(p>0.05):
              temp_cnt = temp_cnt + 1
              sum_v=sum_v+rc1
        sim_score[row][colm]=np.round(sum_v/temp_cnt,3) if(temp_cnt>0) else 1

#Applicability score AUC computation
def Classification_Score(training_data,testing_data):
    X_train=training_data[:,:-1]
    y_train=training_data[:,-1]
    X_test=testing_data[:,:-1]
    y_test=testing_data[:,-1]
    sc=StandardScaler().fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    auc_score=0
    itr=10
    for i in range(itr):
        sc_cls=StandardScaler()
        X_train=sc_cls.fit_transform(X_train)
        X_test=sc_cls.transform(X_test)
        classi1= RandomForestClassifier()
        classi2=LogisticRegression(solver='lbfgs')
        classi3=GaussianNB()
        classi4=svm.SVC(probability=True,gamma='scale')
        classi5=DecisionTreeClassifier()
        classi1.fit(X_train,y_train)
        classi2.fit(X_train,y_train)
        classi3.fit(X_train,y_train)
        classi4.fit(X_train,y_train)
        classi5.fit(X_train,y_train)
        score=np.round((roc_auc_score(y_test, classi1.predict_proba(X_test)[:, 1])+roc_auc_score(y_test, classi2.predict_proba(X_test)[:, 1])+roc_auc_score(y_test, classi3.predict_proba(X_test)[:, 1])+roc_auc_score(y_test, classi4.predict_proba(X_test)[:, 1])+roc_auc_score(y_test, classi5.predict_proba(X_test)[:, 1]))/5,5)
        auc_score = auc_score + score
    return np.round(auc_score/itr,3)

#Phase I: stage2 :Applicability score computation
def Applicability_Score():
    for row in range(len(list1)):
     te=list1[row]
     PATH=os.path.join(r'./content',te)
     Adata=pd.read_csv(PATH)
     Adata=Adata.drop_duplicates(keep='first')
     ta=Adata.values
     ta_x=ta[:,:-1]
     aX_train=ta_x[0,:]
     aXY_train=ta[0,:]
     for tr in list1:
       if(tr!=te):
        PATH=os.path.join(r'./content', tr)
        train=pd.read_csv(PATH)
        train=train.drop_duplicates(keep='first')
        tra=train.values
        tr_x=tra[:,:-1]
        aXY_train=np.vstack((aXY_train,tra))
     aXY_train=aXY_train[1:,:]
     aX_train=aXY_train[:,:-1]
     indices=np.zeros(len(ta_x)*1,dtype=int)
     k=0
     for i in ta_x:
         tmp1=np.array([i])
         temp=cosine_similarity(tmp1,aX_train)
         ind = (-temp[0]).argsort()[:1]
         indices[k:k+1]=ind
         k=k+1
     unique, counts = np.unique(indices, return_counts=True)
     counts1=(-counts).argsort()
     unique1=unique[counts1]
     test_set=aXY_train[unique1]
     total=0
     for colm in range(len(list1)):
       tr=list1[colm]
       if(tr==te):
        apl_score[row][colm]=0
       else:
        PATH=os.path.join(r'./content', tr)
        train=pd.read_csv(PATH)
        tra=train.values
        tra_len=len(tra)
        test_len=math.floor((30*tra_len)/100)
        index=np.array(np.all((tra[:,None,:]==test_set[None,:,:]),axis=-1).nonzero()).T
        total=total+len(index)
        index1=index[:,0]
        training_data = tra
        testing_data = test_set[:test_len,:]
        apl_score[row][colm] = Classification_Score(training_data, testing_data)
"""
#Direct SAPS with selected sources
def classi_SPS_fun(X_train,y_train,X_test,y_test):
    sc=StandardScaler().fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    print(X_train.shape,y_train.shape)
    arr=np.zeros(35)
    cnt=0
    classi1=RandomForestClassifier()
    classi2=LogisticRegression(solver='sag',max_iter=2000)
    classi3=GaussianNB()
    classi4=svm.SVC(probability=True,gamma='scale')
    classi5=DecisionTreeClassifier()
    classi1.fit(X_train,y_train)
    classi2.fit(X_train,y_train)
    classi3.fit(X_train,y_train)
    classi4.fit(X_train,y_train)
    classi5.fit(X_train,y_train)
    y_pred1=classi1.predict(X_test)
    y_pred2=classi2.predict(X_test)
    y_pred3=classi3.predict(X_test)
    y_pred4=classi4.predict(X_test)
    y_pred5=classi5.predict(X_test)
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred1).ravel()
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    arr[cnt:cnt+7]=[FPR, TPR, f1_score(y_test,y_pred1,average="micro"), roc_auc_score(y_test, classi1.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred1),bal]
    cnt=cnt+7
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred2).ravel()
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    arr[cnt:cnt+7]=[FPR, TPR, f1_score(y_test,y_pred2,average="micro"), roc_auc_score(y_test, classi2.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred2),bal]
    cnt=cnt+7
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred3).ravel()
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    arr[cnt:cnt+7]=[FPR, TPR, f1_score(y_test,y_pred3,average="micro"), roc_auc_score(y_test, classi3.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred3),bal]
    cnt=cnt+7
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred4).ravel()
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    arr[cnt:cnt+7]=[FPR, TPR, f1_score(y_test,y_pred4,average="micro"), roc_auc_score(y_test, classi4.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred4),bal]
    cnt=cnt+7
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred5).ravel()
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    arr[cnt:cnt+7]=[FPR, TPR, f1_score(y_test,y_pred5,average="micro"), roc_auc_score(y_test, classi5.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred5),bal]
    return arr
"""
warnings.filterwarnings('ignore')

#Phase I: Source Project selection (SAPS)
########################################

list1=np.array(['ant-1.7.csv','arc.csv', 'camel-1.6.csv','ivy-2.0.csv','jedit-4.2.csv','log4j-1.0.csv','lucene-2.0.csv','poi-2.0.csv','redaktor.csv','synapse-1.2.csv','tomcat.csv','velocity-1.6.csv','xalan-2.4.csv','xerces-1.3.csv'])
#list1=np.array(['CM1.csv','KC3.csv','MC1.csv','MC2.csv','MW1.csv','PC1.csv','PC2.csv','PC3.csv','PC4.csv','PC5.csv'])

sc = StandardScaler()
sc1 = StandardScaler()
sim_score = np.zeros((len(list1),len(list1)))
apl_score = np.zeros((len(list1),len(list1)))
Similarity_Score_WSR()
Applicability_Score()
source_projects_ind = np.zeros((len(list1),len(list1)))
source_projects_ind1 = np.zeros((len(list1),len(list1)))
for kk in range(len(list1)):
  train_sources_ind1 = np.where((1-sim_score[kk])>(mean(1-sim_score[kk])))
  train_sources_ind2 = np.where(apl_score[kk]>(mean(apl_score[kk])))
  sources_ind = np.intersect1d(train_sources_ind1[0],train_sources_ind2[0])
  source_projects_ind[kk,sources_ind] = 1
print("List of source projects for each target Index representation:")
print(source_projects_ind)

#reading target project
for i in range(len(list1)):
  test_ds =list1[i]
  print("target project: ",test_ds)
  PATH=os.path.join(r'./content',test_ds)
  test=pd.read_csv(PATH)
  XY_test=test.values
  #reading combined source projects
  temp = np.where(source_projects_ind[i]==1)
  train_sources = list1[temp[0]] #test dataset sources index
  print("project is: ", test_ds)
  print("Sources are: ", train_sources)
  colc=len(test.columns)
  aXY_train=np.zeros((0,colc))
  for tra in train_sources:
    if(test_ds!=tra):
      PATH=os.path.join(r'./content', tra)
      train=pd.read_csv(PATH)
      XY_train=train.values
      aXY_train=np.vstack((aXY_train,XY_train))
  x_train = aXY_train[:,:-1]
  x_test = XY_test[:,:-1]
  y_train = aXY_train[:,-1]
  y_test = XY_test[:,-1]
  print("SAPS data: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

  #Phase II: Resampling
  #############################
  indm = np.where(aXY_train[:,-1]==0)
  train_maj = aXY_train[indm[0],:-1]
  indmi = np.where(aXY_train[:,-1]==1)
  train_min = aXY_train[indmi[0],:-1]
  maj_len = len(train_maj)
  min_len = len(train_min)
  maj_red = int(maj_len/2)
  min_inc = maj_red-min_len
  #print(maj_len,min_len,maj_red,min_inc)

  #deviding target data into 2 clusters
  model=KMeans(init='k-means++',n_clusters=2)
  clusters=model.fit_predict(XY_test[:,:-1])
  clust1=np.where(clusters == 0)
  clust2=np.where(clusters == 1)
  test_maj,test_min= (x_test[clust1[0]],x_test[clust2[0]]) if len(clust1[0])>len(clust2[0]) else (x_test[clust2[0]],x_test[clust1[0]])
  print(len(test_maj),len(test_min),len(clust1[0]),len(clust2[0]))

  #undersampling source majority data
  temp_list = np.zeros(len(train_maj))
  ii=0
  for i in train_maj:
   for j in test_maj:
    difrn=i-j
    if(difrn.any() == 0):
      temp_list[ii] = temp_list[ii]+1
    else:
      w,p=wilcoxon(difrn)
      temp_list[ii] = temp_list[ii]+1 if(p>0.05) else temp_list[ii]+0
   #print(ii,temp_list[ii])
   ii=ii+1
  train_maj = np.column_stack((train_maj,temp_list))
  indm = np.argsort(train_maj[:,-1])[::-1][:maj_red]
  train_maj = train_maj[indm]
  train_maj = train_maj[:,:-1]
  train_maj_y = np.zeros(len(train_maj)).reshape(-1,1)

  #oversampling source minority data
  train_min
  test_min.shape
  tr_min_len =len(train_min)
  te_min_len = len(test_min)
  maxv = np.amax(test_min,axis=0)
  minv = np.amin(test_min,axis=0)
  new_train_min = np.zeros((min_inc,colc-1))
  cnt=0
  while(cnt<min_inc):
    min1 = random.randint(0,tr_min_len-1)
    min2 = random.randint(0,te_min_len-1)
    randv = np.random.random(colc-1)
    new_train_min[cnt] = (randv*(train_min[min1,:] + test_min[min2]))/(maxv-minv)
    cnt = cnt + 1
  train_min = np.vstack((train_min,new_train_min))
  train_min_y = np.ones(len(train_min)).reshape(-1,1)
  X_train = np.vstack((train_maj,train_min))
  y_train = np.vstack((train_maj_y,train_min_y))
  y_train = y_train[:,-1]
  X_test = XY_test[:,:-1]
  y_test = XY_test[:,-1]
  np.any(np.isnan(X_train))
  print("resampled data: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

  #Phase III: Stacked Autoencoder
  ##########################
  reset_graph()
  d_inputs = colc-1
  d_hidden1 = 10
  d_hidden2 = 4  # codings
  d_hidden3 = d_hidden1
  d_outputs = d_inputs
  n_class = 2

  learning_rate = 0.01
  l2_reg = 0.0005

  initializer = tf.contrib.layers.variance_scaling_initializer()
  activation = tf.nn.elu
  regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

  ##SAE Unsupervised learning
  #Input data
  X = tf.placeholder(tf.float32,shape=[None, d_inputs])
  #Hidden layer1 (first code generating layer)
  weights1_init = initializer([d_inputs, d_hidden1])
  weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
  biases1 = tf.Variable(tf.zeros(d_hidden1), name="biases1")
  hidden1 = activation(tf.matmul(X, weights1) + biases1)
  #Output layer (input reconstruction)
  weights1_ = tf.transpose(weights1,name="weights1_")
  biases1_ = tf.Variable(tf.zeros(d_outputs),name="biases1_")
  outputs1_ = activation(tf.matmul(hidden1,weights1_) + biases1_)
  #Objective function: MSE + L2 penalty
  reconstruction_loss_phase1 = tf.reduce_mean(tf.square(outputs1_ - X))
  reg_loss_phase1 = regularizer(weights1)
  J_phase1 = reconstruction_loss_phase1 + reg_loss_phase1
  optimizer_phase1 = tf.train.AdamOptimizer(learning_rate)
  training_op_phase1 = optimizer_phase1.minimize(J_phase1)

  #SAE supervised learning (also building its own tf subgraph)
  #Input data (intermediate representation)
  Z = tf.placeholder(tf.float32,shape=[None, d_hidden1])
  #Hidden layer2 (second code generating layer)
  weights2_init = initializer([d_hidden1, d_hidden2])
  weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
  biases2 = tf.Variable(tf.zeros(d_hidden2), name="biases2")
  hidden2_ = activation(tf.matmul(Z, weights2) + biases2)
  #Hidden layer3 (second intermediate representation reconstruction layer = output for this graph)
  weights2_ = tf.transpose(weights2,name="weights2_")
  biases2_ = tf.Variable(tf.zeros(d_hidden1),name="biases2_")
  outputs2_ = activation(tf.matmul(hidden2_,weights2_) + biases2_)
  #Objective function: MSE + L2 penalty
  reconstruction_loss_phase2 = tf.reduce_mean(tf.square(outputs2_ - Z))
  reg_loss_phase2 = regularizer(weights2)
  J_phase2 = reconstruction_loss_phase2 + reg_loss_phase2

  optimizer_phase2 = tf.train.AdamOptimizer(learning_rate)
  training_op_phase2 = optimizer_phase2.minimize(J_phase2)

  #Stacking the layers together to build the multilayer autoencoder
  #Reconnecting first coding layer with the second one
  hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
  hidden3 = activation(tf.matmul(hidden2, weights2_) + biases2_)
  #Reconnecting first decoding layer with the output one
  outputs = activation(tf.matmul(hidden3, weights1_) + biases1_)
  #Stacking the layers together to build a classifier
  weights3_init_stack = initializer([d_hidden2, n_class])
  weights3_stack = tf.Variable(weights3_init_stack, dtype=tf.float32, name="weights3_mlp")
  biases3_stack = tf.Variable(tf.zeros(n_class), name="biases3_mlp")
  logit_y = tf.matmul(hidden2, weights3_stack) + biases3_stack
  y = tf.placeholder(tf.int32, shape=[None])
  y_pred_prob = tf.nn.softmax(logit_y)
  y_pred = tf.argmax(y_pred_prob,1)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_y)
  reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3_stack)
  loss = cross_entropy + reg_loss
  optimizer = tf.train.AdamOptimizer(learning_rate)
  training_op = optimizer.minimize(loss)

  batch_size = 50
  n_labeled_instances = len(X_train)
  init = tf.global_variables_initializer()
  saver = tf.train.Saver() #one saver for all sessions
  with tf.Session() as sess:
    n_epochs = 30
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        X_train_tmp=X_train
        for iteration in range(n_batches):
            #print("epoch--batch: ", epoch, iteration)
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch= X_train_tmp[:batch_size]
            X_train_temp=X_train_tmp[batch_size:]
            sess.run(training_op_phase1, feed_dict={X: X_batch})
        if(X_train_tmp.size != 0):
            sess.run(training_op_phase1, feed_dict={X: X_train_tmp})
        ae1_loss = J_phase1.eval(feed_dict={X: X_train})
        print(ae1_loss)
    Z_train = sess.run(hidden1,feed_dict={X: X_train})  #compressed data Z_train of x_train

    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        Z_train_tmp=Z_train
        for iteration in range(n_batches):
            #print("epoch--batch: ", epoch, iteration)
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            #indices = np.random.permutation(len(y_train))[:batch_size]
            Z_batch = Z_train_tmp[:batch_size]
            Z_train_tmp = Z_train_tmp[batch_size:]
            sess.run(training_op_phase2, feed_dict={Z: Z_batch})
        if(Z_train_tmp.size != 0):
            sess.run(training_op_phase2, feed_dict={Z: Z_train_tmp})
        ae2_loss = J_phase2.eval(feed_dict={Z: Z_train})
        print(ae2_loss)
    Z1_train = sess.run(hidden2,feed_dict={X: X_train})
    print("")
    print("---------------------------------------------------------------------------------")
    print("reconstructed input")
    results = sess.run(hidden2,feed_dict={X: X_test})
    print("SAE text data output -- len(results)")

    #Supervised learning
    n_epochs = 10
    weights1.eval()
    weights2.eval()
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        X_train_tmp=X_train
        indices = np.random.permutation(n_labeled_instances)[:batch_size]
        for iteration in range(n_batches):
            #print("epoch--batch: ", epoch, iteration)
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = X_train_tmp[indices], y_train[indices]
            indices=indices[batch_size:]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})  #for weight optimization
        if(len(indices) != 0):
            X_batch, y_batch = X_train_tmp[indices], y_train[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        correct_prediction = tf.equal(tf.argmax(y_pred_prob,1), tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("accuracy", sess.run(accuracy, feed_dict={X: X_test, y: y_test}))
        prediction2 = y_pred_prob
        y_test_predicted2 = prediction2.eval(feed_dict={X: X_test}, session=sess)
        prediction = tf.argmax(y_pred_prob,1)   #convert probabilities of [0.1 0.9] to 0/1
        y_test_predicted = prediction.eval(feed_dict={X: X_test}, session=sess)
        print(y_test_predicted)
        print(y_test)
        cm=confusion_matrix(y_test, y_test_predicted)
        print(cm)
        tn=cm[0][0]
        fp=cm[0][1]
        fn=cm[1][0]
        tp=cm[1][1]
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        FPR = (fp/(fp+tn))
        roc = roc_auc_score(y_test,y_test_predicted)
        g_mean = math.sqrt(TPR*TNR)
        bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
        print(FPR,TPR,roc,g_mean,matthews_corrcoef(y_test,y_test_predicted),bal)
