import numpy as np

class_count = np.array([326256,135005,109385,188598,482812,239672])
class_count_n=(class_count-np.min(class_count))/(np.max(class_count)-np.min(class_count))
weights=1/(1+class_count_n)



correct=np.array([3249866.,756220.,532575.,758461.,1661403.,779487.])
count=np.array([3996461.,930312.,669844.,919657.,2047213.,1190313.])

accuracy=correct/count
acc_inv=np.power(1-accuracy,3)
#accuracy_n=(accuracy-np.min(accuracy))/(np.max(accuracy)-np.min(accuracy))
acc_inv=np.max(count)-count*0.4
accuracy_s=acc_inv/np.sum(acc_inv)
print(accuracy)
print(acc_inv)
print(accuracy_s)
print(np.sum(accuracy_s))