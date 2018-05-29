import inspect
import re
import sys
def prints(x,fname="debug"):
	#print("[@"+sys._getframe().f_code.co_name+"]")
	frame = inspect.currentframe().f_back
	s = inspect.getframeinfo(frame).code_context[0]
	r = re.search(r"\((.*)\)", s).group(1)
	print("[@"+fname+"] "+"{} = {}".format(r,x))

#x=34
if __name__ == "__main__":
	d={}
	d["f"]=[23,3]
	prints(d["f"])
