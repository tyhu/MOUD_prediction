import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from util_ml import *
from sklearn.metrics import accuracy_score
from q_learning import *

def statefunc(currenttime,score,currpred,endbool):
    score = abs(score)
    if endbool:
        state = [0,0,0,1]
        return tuple(state)
    else:
        if score<0.5: scoreidx = 0
        elif score<1.5: scoreidx = 1
        else: scoreidx = 2
        state = [currenttime,scoreidx,currpred,0]
        return tuple(state)


labfn = '/home2/tingyaoh/sentiment/MOUD/MultimodalFeatures/audioFeatures/cats.txt'
lab = [l.split()[1] for l in file(labfn)]
y = (np.array(lab)=='positive').astype('int')

fnlst = [l.split()[0] for l in file(labfn)]
xlst = []
for fn in fnlst:
    x = np.genfromtxt('opensmile_feat/'+fn+'_seq.feat',delimiter=';')
    xlst.append(x.T.tolist())
X = np.array(xlst).astype('float')
### X.shape: (datanum,featnum,prediction_step_num)

np.random.seed(1234)
X, y = RandomPerm(X,y)

pred_sec_lst = [1,2,3,4,5]
totaltime = 0
step_cost = 0
ytest_total = []
ypred_total = []
ytestlst = []
yfinallst = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    ### collecting mdp history
    historylst = []
    for Xtrtr, ytrtr, Xtrts, ytrts in KFold(Xtrain,ytrain,5):
        predlst = []
        scorelst = []
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            clf = LinearSVC(C=0.01,penalty='l1',dual=False)
            #clf.fit(Xtrtr[:,:,pred_idx],ytrtr)
            clf.fit(Xtrtr[:,:,-1],ytrtr)
            ytrpred = clf.predict(Xtrts[:,:,pred_idx])
            predlst.append(ytrpred.tolist())
            scores = clf.decision_function(Xtrts[:,:,pred_idx])
            scorelst.append(scores.tolist())
        ytrpreds = np.array(predlst)
        ytrscore = np.array(scorelst)
        #print ytrscore
        #print ytrpreds.shape, ytrscore.shape, ytrts.shape

        prednum = ytrpreds.shape[0]
        datanum = ytrpreds.shape[1]
        for i in range(datanum):
            for pred_idx, pred_sec in enumerate(pred_sec_lst):
                currenttime = pred_sec
                score = ytrscore[pred_idx,i]
                currpred = ytrpreds[pred_idx,i]
                answer = ytrts[i]
                endbool = 0
                if pred_idx==prednum-1 or Xtrts[i,-1,pred_idx]==Xtrts[i,-1,pred_idx+1]:
                    endbool = 1
                if answer==currpred: reward = 1
                else: reward = -1
                state = statefunc(currenttime,score,currpred,endbool)
                if state[-1]==1:
                    action = 'y'
                    tup = (state,action,reward,None)
                    historylst.append(tup)
                    continue
                nexttime = pred_sec_lst[pred_idx+1]
                nextscore = ytrscore[pred_idx+1,i]
                nextpred = ytrpreds[pred_idx+1,i]
                nextend = 0
                if pred_idx==prednum-2 or Xtrts[i,-1,pred_idx+1]==Xtrts[i,-1,pred_idx+2]:
                    nextend = 1
                state2 = statefunc(nexttime,nextscore,nextpred,nextend)
                action = 'y'
                tup = (state,action,reward,None)
                historylst.append(tup)
                action = 'n'
                tup = (state,action,-step_cost,state2)
                historylst.append(tup)
    
    ### end of collecting mdp history
    ### models training
    print 'model training'
    clflst = []
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        clf = LinearSVC(C=0.01,penalty='l1',dual=False)
        #clf.fit(Xtrain[:,:,pred_idx],ytrain)
        clf.fit(Xtrain[:,:,-1],ytrain)
        clflst.append(clf)

    ### mdp training
    print 'mdp training'
    mdp = myMDP(alpha = 0.5, gamma=0.9, iternum = 500)
    mdp.Init_From_History(historylst)
    mdp.QLearn(historylst)
    #mdp.PrintPolicy()

    ### testing
    print 'testing'
    ypredlst = []
    yscorelst = []
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        ypredlst.append(clflst[pred_idx].predict(Xtest[:,:,pred_idx]).tolist())
        yscorelst.append(clflst[pred_idx].decision_function(Xtest[:,:,pred_idx]))
    yfinal = np.zeros_like(ytest)
    datanum = yfinal.shape[0]
    for i in range(datanum):
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            currtime = pred_sec
            score = yscorelst[pred_idx][i]
            currpred = ypredlst[pred_idx][i]
            endbool = 0
            #if pred_idx==prednum-1 or Xtest[i,-1,pred_idx+1]==Xtest[i,-1,pred_idx+2]:
            if pred_idx==prednum-1 or Xtest[i,-1,pred_idx]==Xtest[i,-1,pred_idx+1]:
                endbool = 1
            state = statefunc(currtime, score, currpred, endbool)
            #if pred_sec==6 or endbool:
            if mdp.Policy(state)=='y':
                yfinal[i] = currpred
                totaltime+=currtime
                print 'time: ',currtime
                print state
                break
            
    yfinallst+=yfinal.tolist()
    ytestlst+=ytest.tolist()
print 'totaltime: ',totaltime
print accuracy_score(yfinallst,ytestlst)

