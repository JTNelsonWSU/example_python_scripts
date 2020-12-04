#############################For regressions############# 
#####The following script is responsible for generating multiple linear regressions for all SNPs and environmental data. The output should be a df with slope, intercept, Rvalue, pvalue, R2, and standard error


import pandas as pd
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np




sig = pd.read_csv("ag1000G.phase2.allele.frequencies.txt")

sig = sig.set_index("Population/location")
sig.index.names = [None]


sig_MaxTempWarm = sig.drop(["N","Lats.","Lons.","srad","MinTempCold","AnnPre","Iso","TempSea","PreDry","AnnTemp","PreWet","vapr"],axis=1)
sig_MaxTempWarm.head()

sig_MaxTempWarm = sig_MaxTempWarm.astype(float)


sig_var_SNPs = pd.DataFrame(columns=[0,1,2,3,4])

for i in range(0,len(sig_MaxTempWarm.columns)):
    temp = (linregress(sig_MaxTempWarm["MaxTempWarm"],sig_MaxTempWarm.iloc[:,i]))
    temp = pd.DataFrame(temp)
    transposed = temp.transpose()
    sig_var_SNPs = sig_var_SNPs.append(transposed, ignore_index=True)


sig_var_SNPs.shape
final_sig_MaxTempWarm = sig_var_SNPs
final_sig_MaxTempWarm.columns = ["Slope","Intercept","Rvalue","Pvalue","Stderr"]
final_sig_MaxTempWarm["R2"] = final_sig_MaxTempWarm["Rvalue"]**2
final_sig_MaxTempWarm = final_sig_MaxTempWarm.drop(0)
final_sig_MaxTempWarm.head()

final_sig_MaxTempWarm.to_csv("final.sig.2R.GEA.inversion.MaxTempWarm.txt", index = None, sep = "\t")




