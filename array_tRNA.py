from sklearn import datasets
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.svm import SVR
import numpy as np

#导入文件数据
filename4 = 'E:/ML_data/Genome_tRNA_features.txt'
filename5 = 'E:/ML_data/spices.txt'
f4 = open(filename4,'r')
f5 = open(filename5,'r')


speices = []
all_speices = {}
data = []
lists = []


data_tRNA = f4.readlines()
species_count = len(data_tRNA)

for i in range(1,species_count):
    tmp_data = []
    speices.append(data_tRNA[i].split('\t')[1].strip())

    tmp_data.extend([float(j) for j in data_tRNA[i].split('\t')[2:]if j != '\n'])
    lists.append(tmp_data)

X = np.array(lists)

for i in f5.readlines():
    all_speices[i.split('\t')[0]] = float(i.split('\t')[1].strip())
y = [all_speices[i] for i in speices]

f4.close()
f5.close()

best_sc = 100000
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
#进行参数的不断调整，找到参数使得均方根误差最小
for gamma in [0.01,0.1,1,10,100]:
    for c in [0.01,0.1,1,10,100]:
        svr = SVR(kernel = 'rbf',gamma = gamma,C = c)
        #cv = 10 表示使用10折交叉验证法
        sc = np.sqrt(-cross_val_score(svr,X_train,y_train,cv = 10,scoring = "neg_mean_squared_error"))
        sc = sc.mean()
        if sc < best_sc:
            best_sc = sc
            best_parameters = {'gamma':gamma,"C":c}

svr = SVR(kernel = 'rbf',gamma = best_parameters['gamma'],C = best_parameters['C'])
svr.fit(X_train,y_train)


