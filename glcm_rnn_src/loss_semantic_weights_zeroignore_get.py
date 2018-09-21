import numpy as np
correct=np.array([158235., 22133., 15153., 31739., 68290., 33439.])
count=np.array([212136., 42284., 32426., 44510., 103006., 58038.])

accuracy=np.divide(correct,count)

acc_inv=np.power(1-accuracy,3)
#accuracy_n=(accuracy-np.min(accuracy))/(np.max(accuracy)-np.min(accuracy))
acc_inv=np.max(count)-count*0.4



acc_inv=[0, 127281.6, 195222.4, 199165.6, 194332.,  170933.6, 188920.8]
accuracy_s=acc_inv/np.sum(acc_inv)
print(accuracy)
print(acc_inv)
print(accuracy_s)
print(np.sum(accuracy_s))

