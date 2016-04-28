import sys
sys.path.append('/home2/tingyaoh/ML/PyAI/reinforcement')
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from util_ml import *
from sklearn.metrics import accuracy_score
from fa_q_learning import *

def statefunc(currenttime,score,currpred,endbool):
    score = abs(score)
    if endbool:
        state = [0,0,0,1]
        return tuple(state)
    else:
        if score<0.5: scoreidx = 0
        elif score<1.0: scoreidx = 1
        else: scoreidx = 2
        state = [currenttime,scoreidx,currpred,0]
        return tuple(state)


pred_sec_lst = range(1,11)
xlst = []
for pred_sec in pred_sec_lst:
    featfn = 'video_feat_'+str(pred_sec)+'.txt'
    feat = np.genfromtxt(featfn,delimiter=',',skip_header=0).astype('float')
    y = np.array(feat[:,0].tolist())
    y = y[np.where(y!=0)]
    x = feat[:,1:]
    x = x[np.where(y!=0)]
    x = np.c_[x,np.zeros(x.shape[0]),np.zeros(x.shape[0]),np.zeros(x.shape[0])]
    xlst.append(x.tolist())
X = np.array(xlst)
X = np.swapaxes(X,0,2)
X = np.swapaxes(X,0,1)
### X.shape: (datanum,featnum,prediction_step_num)

### add end tag
pred_sec_lst = [1,2,3,4,5]
for idx in range(X.shape[0]):
    for jdx in range(X.shape[2]):
        X[idx,-3,jdx] = jdx+1
        if jdx==X.shape[2]-1 or np.array_equal(X[idx,:,jdx],X[idx,:,jdx+1]):
            X[idx,-1,jdx] = 100
        if jdx>=pred_sec_lst[-1]-1:
            X[idx,-2,jdx] = 100

#print X[0,:,:]

np.random.seed(1234)
X, y = RandomPerm(X,y)

#pred_sec_lst = [1,2,3,4,5,6,7,8,9,10]
totaltime = 0
step_cost = 1
ytest_total = []
ypred_total = []
ytestlst = []
yfinallst = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    ### collecting mdp history
    state_mat, state2_mat, act_vec, reward_vec = np.array([]),np.array([]),[],[]
    trdatanum = Xtrain.shape[0]
    for idx in range(trdatanum):
        for pred_sec in pred_sec_lst:
            ### if utterance end reached
            if Xtrain[idx,-1,pred_sec-1]==1 or Xtrain[idx,-2,pred_sec-1]==1:
                for act in ['0','1','2']:
                    act_vec.append(act)
                    state_mat = np.vstack([state_mat,Xtrain[idx,:,pred_sec-1]]) if state_mat.size else Xtrain[idx,:,pred_sec-1]
                    state2_mat = np.vstack([state2_mat,np.zeros(Xtrain.shape[1])]) if state2_mat.size else np.zeros(Xtrain.shape[1])
                    if act=='0': reward_vec.append(-100)
                    elif ytrain[idx]==int(act): reward_vec.append(1.0)
                    else: reward_vec.append(-1.0)
                break
            else:
                for act in ['0','1','2']:
                    act_vec.append(act)
                    state_mat = np.vstack([state_mat,Xtrain[idx,:,pred_sec-1]]) if state_mat.size else Xtrain[idx,:,pred_sec-1]
                    state2_mat = np.vstack([state2_mat,Xtrain[idx,:,pred_sec]]) if state2_mat.size else Xtrain[idx,:,pred_sec]
                    if act=='0': reward_vec.append(step_cost)
                    elif ytrain[idx]==int(act): reward_vec.append(1.0)
                    else: reward_vec.append(-1.0)
    
    ### end of collecting mdp history
    act_vec, reward_vec = np.array(act_vec), np.array(reward_vec)

    ### mdp training
    print 'mdp training'
    mdp = FA_MDP(alpha = 0.0001, gamma=0.9, iternum = 1000)
    mdp.Init_From_Lsts(['1','2','0'],48) # 0 means not sure
    mdp.BatchQLearn(state_mat,act_vec,reward_vec,state2_mat,ld=1)
    #mdp.PrintPolicy()

    ### testing
    print 'testing'
    tsdatanum = Xtest.shape[0]
    for idx in range(tsdatanum):
        for jdx in range(len(pred_sec_lst)):
            tidx = pred_sec_lst[jdx]-1
            act = mdp.Policy(Xtest[idx,:,tidx])
            if act!='0':
                print act, pred_sec_lst[jdx], ytest[idx]
                break
        if act=='0':
            qvec = mdp.GetQValues(Xtest[idx,:,tidx])
            if qvec[0]>qvec[1]: act='1'
            else: act='2'
        yfinallst.append(int(act))
        ytestlst.append(int(ytest[idx]))

print yfinallst
print ytestlst
print accuracy_score(yfinallst,ytestlst)


