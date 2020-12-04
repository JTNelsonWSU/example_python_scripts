#####Converts allele count data to BayPass input for association analysis

import pandas as pd

##########
GHM_1 = pd.read_csv("GHM_1_X.ids.thin10kb.X.matrix.5.8.frq.count.noheader.txt",sep="\t",header=None)
ref_list = []
alt_list = []
temp_ref = range(1,567*2,2)
temp_alt = range(2,568*2,2)

for i in temp_ref:
	ref_list.append(i)


for i in temp_alt:
	alt_list.append(i)

ref_list = pd.DataFrame(ref_list)
alt_list = pd.DataFrame(alt_list)
GHM_1ref = pd.concat([GHM_1[4], ref_list],axis=1)
GHM_1alt = pd.concat([GHM_1[5], alt_list],axis=1)
GHM_1ref.columns = ['allele', 'seq']
GHM_1alt.columns = ['allele', 'seq']
GHM_1_all = pd.concat([GHM_1ref,GHM_1alt],axis=0)
GHM_1_all = GHM_1_all.sort_values('seq')



GHM_2 = pd.read_csv("GHM_2_X.ids.thin10kb.X.matrix.5.8.frq.count.noheader.txt",sep="\t",header=None)
ref_list = []
alt_list = []
temp_ref = range(1,567*2,2)
temp_alt = range(2,568*2,2)

for i in temp_ref:
	ref_list.append(i)


for i in temp_alt:
	alt_list.append(i)

ref_list = pd.DataFrame(ref_list)
alt_list = pd.DataFrame(alt_list)
GHM_2ref = pd.concat([GHM_2[4], ref_list],axis=1)
GHM_2alt = pd.concat([GHM_2[5], alt_list],axis=1)
GHM_2ref.columns = ['allele', 'seq']
GHM_2alt.columns = ['allele', 'seq']
GHM_2_all = pd.concat([GHM_2ref,GHM_2alt],axis=0)
GHM_2_all = GHM_2_all.sort_values('seq')



GHM_3 = pd.read_csv("GHM_3_X.ids.thin10kb.X.matrix.5.8.frq.count.noheader.txt",sep="\t",header=None)
ref_list = []
alt_list = []
temp_ref = range(1,567*2,2)
temp_alt = range(2,568*2,2)

for i in temp_ref:
	ref_list.append(i)


for i in temp_alt:
	alt_list.append(i)

ref_list = pd.DataFrame(ref_list)
alt_list = pd.DataFrame(alt_list)
GHM_3ref = pd.concat([GHM_3[4], ref_list],axis=1)
GHM_3alt = pd.concat([GHM_3[5], alt_list],axis=1)
GHM_3ref.columns = ['allele', 'seq']
GHM_3alt.columns = ['allele', 'seq']
GHM_3_all = pd.concat([GHM_3ref,GHM_3alt],axis=0)
GHM_3_all = GHM_3_all.sort_values('seq')




GHS_1 = pd.read_csv("GHS_1_X.ids.thin10kb.X.matrix.5.8.frq.count.noheader.txt",sep="\t",header=None)
ref_list = []
alt_list = []
temp_ref = range(1,567*2,2)
temp_alt = range(2,568*2,2)

for i in temp_ref:
	ref_list.append(i)


for i in temp_alt:
	alt_list.append(i)

ref_list = pd.DataFrame(ref_list)
alt_list = pd.DataFrame(alt_list)
GHS_1ref = pd.concat([GHS_1[4], ref_list],axis=1)
GHS_1alt = pd.concat([GHS_1[5], alt_list],axis=1)
GHS_1ref.columns = ['allele', 'seq']
GHS_1alt.columns = ['allele', 'seq']
GHS_1_all = pd.concat([GHS_1ref,GHS_1alt],axis=0)
GHS_1_all = GHS_1_all.sort_values('seq')

temp_input = pd.concat([GHM_1_all['allele'],GHM_2_all['allele'],GHM_3_all['allele'],GHS_1_all['allele']])

temp_input.to_csv('temp.BayPass.input.X.txt', index=False, header=False,sep=" ")

