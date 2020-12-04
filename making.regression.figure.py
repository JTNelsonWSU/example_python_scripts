#Extracted from jupyter notebook. Used to compare the slope, R2 and p-value of mulitple linear regressions for genetic and environmental data.
####Makes several joined distributions of regression results for two different environmental predictors

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import pandas as pd
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


annpre2L = pd.read_csv("final.sig.2L.GEA.inversion.AnnPre.txt", sep="\t")
mintempcold2L = pd.read_csv("final.sig.2L.GEA.inversion.MinTempCold.txt", sep="\t")
annpre2L.shape
mintempcold2L.shape



# In[ ]:


########Making a multi-histogram figure to compare our results
####For Slope
#plt.hist(final_null_predry["Slope"],bins=50, alpha=0.5, label="Null")
plt.hist(annpre2L["Slope"],bins=100, alpha=0.8, label="2L")
plt.hist(mintempcold2L["Slope"],bins=200, alpha=0.5, label="2L")
plt.show()

####For R2
#plt.hist(final_null_predry["R2"],bins=50, alpha=0.5, label="Null")
plt.hist(annpre2L["R2"],bins=50, alpha=0.8, label="2L")
plt.hist(mintempcold2L["R2"],bins=50, alpha=0.5, label="2L")
plt.show()

####For Pvalue
#plt.hist(final_null_predry["Pvalue"],bins=100, alpha=0.5, label="Null")
plt.hist(annpre2L["Pvalue"],bins=1, alpha=0.8, label="2L")
plt.hist(mintempcold2L["Pvalue"],bins=50, alpha=0.5, label="2L")
plt.show()


# In[ ]:





# In[ ]:


############for 2R


# In[ ]:


vapr2R = pd.read_csv("final.sig.2R.GEA.inversion.Vapr.txt", sep="\t")
predry2R = pd.read_csv("final.sig.2R.GEA.inversion.PreDry.txt", sep="\t")
maxtempwarm2R = pd.read_csv("final.sig.2R.GEA.inversion.MaxTempWarm.txt", sep="\t")




# In[ ]:





# In[ ]:






# In[ ]:


########Making a multi-histogram figure to compare our results
####For Slope
#plt.hist(final_null_predry["Slope"],bins=50, alpha=0.5, label="Null")
plt.hist(vapr2R["Slope"],bins=50, alpha=0.8, label="2L")
plt.hist(predry2R["Slope"],bins=50, alpha=0.5, label="2L")
#plt.hist(maxtempwarm2R["Slope"],bins=50, alpha=0.5, label="2L")
plt.show()

####For R2
#plt.hist(final_null_predry["R2"],bins=50, alpha=0.5, label="Null")
plt.hist(vapr2R["R2"],bins=50, alpha=0.8, label="2L")
plt.hist(predry2R["R2"],bins=50, alpha=0.5, label="2L")
#plt.hist(maxtempwarm2R["R2"],bins=50, alpha=0.5, label="2L")
plt.show()

####For Pvalue
#plt.hist(final_null_predry["Pvalue"],bins=100, alpha=0.5, label="Null")
plt.hist(vapr2R["Pvalue"],bins=50, alpha=0.8, label="2L")
plt.hist(predry2R["Pvalue"],bins=50, alpha=0.5, label="2L")
#plt.hist(maxtempwarm2R["Pvalue"],bins=50, alpha=0.5, label="2L")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




