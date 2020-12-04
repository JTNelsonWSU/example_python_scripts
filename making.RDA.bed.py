#########This script is responsible for converting candidate outliers into .bed format for BedTools Intersection and downstream analyses.

import pandas as pd
import numpy as np


cand1 = pd.read_csv("ag1000G.RDA1.X.cand.predictors.updated.txt",sep=" ")
cand2 = pd.read_csv("ag1000G.RDA2.X.cand.predictors.updated.txt",sep=" ")
cand3 = pd.read_csv("ag1000G.RDA3.X.cand.predictors.updated.txt",sep=" ")
cand4 = pd.read_csv("ag1000G.RDA4.X.cand.predictors.updated.txt",sep=" ")
cand5 = pd.read_csv("ag1000G.RDA5.X.cand.predictors.updated.txt",sep=" ")

cand1 = cand1[['Position','predictor']]
cand1['End'] = cand1['Position'] + 1
cand1['RDA'] = 1
cand1['chr'] = "chrX"
cand1 = cand1[['chr','Position','End','predictor','RDA']]


cand2 = cand2[['Position','predictor']]
cand2['End'] = cand2['Position'] + 1
cand2['RDA'] = 2
cand2['chr'] = "chrX"
cand2 = cand2[['chr','Position','End','predictor','RDA']]


cand3 = cand3[['Position','predictor']]
cand3['End'] = cand3['Position'] + 1
cand3['RDA'] = 3
cand3['chr'] = "chrX"
cand3 = cand3[['chr','Position','End','predictor','RDA']]


cand4 = cand4[['Position','predictor']]
cand4['End'] = cand4['Position'] + 1
cand4['RDA'] = 4
cand4['chr'] = "chrX"
cand4 = cand4[['chr','Position','End','predictor','RDA']]


cand5 = cand5[['Position','predictor']]
cand5['End'] = cand5['Position'] + 1
cand5['RDA'] = 5
cand5['chr'] = "chrX"
cand5 = cand5[['chr','Position','End','predictor','RDA']]


cand1.to_csv("RDA1.updated.X.bed",header=None,index=None,sep="\t")
cand2.to_csv("RDA2.updated.X.bed",header=None,index=None,sep="\t")
cand3.to_csv("RDA3.updated.X.bed",header=None,index=None,sep="\t")
cand4.to_csv("RDA4.updated.X.bed",header=None,index=None,sep="\t")
cand5.to_csv("RDA5.updated.X.bed",header=None,index=None,sep="\t")
