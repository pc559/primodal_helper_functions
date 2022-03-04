from subprocess import call,check_output
import numpy as np

def leggauss_c(n):
    try:
    	res = check_output(["../leggauss_c",str(n)])
    except:
    	res = check_output(["./leggauss_c",str(n)])
    res = str(res)[2:-1]
    res = res.split('\\n')[:-1]
    res = [[float(s) for s in r.split()] for r in res]
    res = np.array(res)
    xs,ws = res.T
    return xs,ws

