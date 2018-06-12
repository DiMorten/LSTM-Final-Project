import numpy as np

class_count = np.array([326256,135005,109385,188598,482812,239672])
class_count_n=(class_count-np.min(class_count))/(np.max(class_count)-np.min(class_count))
weights=1/(1+class_count_n)