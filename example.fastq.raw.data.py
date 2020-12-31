######This is an example script for extracting and analyzing raw fastq data.


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from collections import Counter



import io
import glob
#!pip install Bio
from Bio import SeqIO

import os
import math
import pylab as plt
import matplotlib.patches as patches



#Directory location of FASTQ files

fastq_directory_R1 = './S1_S1_R1_001.fastq'
fastq_directory_R2 = './S1_S1_R2_001.fastq'

#filenames = sorted(glob.glob('*.fastq'))
#reads = []
#for file in filenames:
#    single_file = pd.read_csv(file, index_col=None, header=None)
#    reads.append(single_file)




#################
# Write your work for task #1 here
def read_fastq_F(read1,read2,N): #start function with three paramerters: read1, read2, and N (top Nth lines)
  temp1 = pd.read_csv(read1,header=None) #Read in read1
  temp2 = pd.read_csv(read2,header=None) #Read in read2
  slice_temp1 = temp1.iloc[1::4,:].rename(columns={0:"read1"}).head(N) #walks down rows of a df and grabs every 4 indicies stating with index 1. Renames column and grabs to the Nth row. For read1
  slice_temp2 = temp2.iloc[1::4,:].rename(columns={0:"read2"}).head(N) #walks down rows of a df and grabs every 4 indicies stating with index 1. Renames column and grabs to the Nth row. For read2
  concat_reads = pd.concat([slice_temp1, slice_temp2],axis = 1) #concatenates the sampled reads by columns.
  return concat_reads #returns the local variable "concat_reads" to a global variable for downstream
#################


#################
#Write your work for task #2 here
read1_bar = pd.DataFrame() #makes an empty df
read2_bar = pd.DataFrame() #makes an empty df
length = slice(15) #Sets a slicer that takes the first 15 instances of a string 
for i in range(0,len(df1)): #starts single loop that has len(df1) iterations
    temp1 = pd.DataFrame(io.StringIO(df1.iloc[i,0][length])) #converts read1 in coloum 0 row i to a string and applies the slicer. Then converts back to a df.
    read1_bar = read1_bar.append(temp1, ignore_index=True) #appends the temp1 to read_bar1 
    temp2 = pd.DataFrame(io.StringIO(df1.iloc[i,1][length])) #converts read2 in coloum 1 row i to a string and applies the slicer. Then converts back to a df.
    read2_bar = read2_bar.append(temp2, ignore_index=True) #appends the temp1 to read_bar1 
    df2 = pd.concat([read1_bar,read2_bar],axis = 1) #combines the barcodes from each read into one df
df2.columns = ["read1","read2"] #changes the columns names of df2
barcodes = pd.read_csv("barcodes_run22.csv", sep = ",") #Loads the barcode data from barcodes_run22.csv
#############

#################
# Write your work for task #3 here
df2 = df2.rename(columns={"read1":"Barcode"}) #Renames the read1 column to Barcode
temp = pd.merge(df2,barcodes,on="Barcode",how="inner") #Performs an inner merge between df2 and barcodes. This is for read1.
temp = temp.rename(columns={"Barcode":"read1","read2":"Barcode"}) #Renames colmns in temp so that read2 is now labled as Barcode.
temp2 = pd.merge(temp,barcodes,on="Barcode",how="inner") #Performs an inner merge between df2 and barcodes. This is for read2.
df3 = temp2[["read1","Barcode"]].rename(columns={"Barcode":"read2"}) #Pulls the first two columns of temp to and renames them "read1" and "read2".
#################


#################
# Write your work for task #4 here
df3['counts'] = 1  #creates an additional column that has interger "1" in every row. Every event (protein interaction) is equal to 1
df4 = pd.pivot_table(df3,values='counts', index=['read2'],columns=['read1'],aggfunc=np.sum,fill_value=0)  # Create a matrix that sums the number of times an MATa barcode is paired with  a MATalpha barcoode. No match = 0.
df4.columns.name = None  #resets the column names
df4.index.name = None #resets the index names
##################

#################
# Write your work for task #5 here
##Creates a heatmap for barcode pairs

fig, ax = plt.subplots(figsize=(11,9))  #makes an empty ploot with x and y dimensions
sns.heatmap(df4, cmap="Blues", vmin=0,linewidth=0.3,linecolor="Black",center=0,annot=True,fmt="d") #creates a heatmap (blue gradient), values are cetered and annoted.
plt.show  #shows heat map
###########

####for making master_read_data df####
#The following lines make a master file that can generate a heatmap for "Barcode", "Strain", "Name", "Description" pairs
temp = df3 #df3 saved as temp
temp = temp.rename(columns={"read1":"Barcode"}) #Renames "read1" with "Barcodes"
read1 = pd.merge(temp,barcodes,on="Barcode", how="inner") #Inner join to find barcode matches for read1
read1 = read1.rename(columns={"Barcode":"read1","read2":"Barcode"}) #Renames "read2" with "Barcodes" and reverts the original read1 header
matster_read_data = pd.merge(read1,barcodes,on="Barcode",how="inner") #Inner join to find barcode matches for read2
matster_read_data.columns = ["read1","read2","counts","Strain_1","Name_1", "Description_1", "Strain_2", "Name_2", "Description_2"] #Renames all columns
#########

#########For strain#####
strain = matster_read_data[["Strain_1","Strain_2","counts"]] #save master_read_data as strain
strain = pd.pivot_table(strain,values='counts',index=['Strain_2'],columns=['Strain_1'], aggfunc=np.sum,fill_value=0) # Create a matrix that sums the number of times an MATa barcode is paired with  a MATalpha barcoode. No match = 0.
strain.columns.name = None #resets the column names
strain.index.name = None #resets the index names
fig, ax = plt.subplots(figsize=(11,9)) #makes an empty ploot with x and y dimensions
sns.heatmap(strain, cmap="Blues",vmin=0, linewidths=0.3, linecolor="Black",center=0,annot=True,fmt="d") #creates a heatmap (blue gradient), values are cetered and annoted.
plt.show() #shows heat map
##########

#########For name
name = matster_read_data[["Name_1","Name_2","counts"]] #save master_read_data as name
name = pd.pivot_table(name,values='counts',index=['Name_2'],columns=['Name_1'], aggfunc=np.sum,fill_value=0) # Create a matrix that sums the number of times an MATa barcode is paired with  a MATalpha barcoode. No match = 0.
name.columns.name = None #resets the column names
name.index.name = None #resets the index names
fig, ax = plt.subplots(figsize=(11,9)) #makes an empty ploot with x and y dimensions
sns.heatmap(name, cmap="Blues",vmin=0, linewidths=0.3, linecolor="Black",center=0,annot=True,fmt="d") #creates a heatmap (blue gradient), values are cetered and annoted.
plt.show() #shows heat map
###########

#########For description
description = matster_read_data[["Description_1","Description_2","counts"]] #save master_read_data as description
description = pd.pivot_table(description,values='counts',index=['Description_2'],columns=['Description_1'], aggfunc=np.sum,fill_value=0) # Create a matrix that sums the number of times an MATa barcode is paired with  a MATalpha barcoode. No match = 0.
description.columns.name = None #resets the column names 
description.index.name = None #resets the inndex names
fig, ax = plt.subplots(figsize=(11,9)) #makes an empty ploot with x and y dimensions
sns.heatmap(description, cmap="Blues",vmin=0, linewidths=0.3, linecolor="Black",center=0,annot=True,fmt="d") #creates a heatmap (blue gradient), values are cetered and annoted.
plt.show() #shows heat map
###########


#####For 6G
print(df3['read1'].value_counts())
print('\n---') 
print(df3['read2'].value_counts())
df3['read_1_2'] = df3['read1'] + "_" + df3['read2']
temp = pd.DataFrame(df3['read_1_2'].value_counts())
print('\n---') 
print(df3['read_1_2'].value_counts())
temp.index.name = "new"
temp.reset_index(inplace=True)
temp.columns = ['read_1_2', "counts"]
print('\n---')
print("Average number of binding events: " + str(temp['counts'].mean()))
print("Minimum: " + str(temp['counts'].min()))
print("Maximum: " + str(temp['counts'].max()))
print('\n---')
plt.hist(temp['counts'],bins=20)
plt.show
plt.xlabel('Counts')
plt.ylabel('Density')
#############


#(6A) Create a second figure of your choice (heatmap, bar plot, scatter plot, etc) to visualize the entire dataset or a portion it. Justify your design choices.
######## The following lines geneate summary stats and three figures for quality Phred Scores
###The following function is responsible for making a df that will contain the phred score for each basepair or read 1 or read 2
def getting_fastq_qualities(filename, ax=None, limit=500000): #Makes a function that takes three arguments: fastq file, "None", and a sequence limit.

    fastq_parser = SeqIO.parse(filename, "fastq")  #opens the fastq file
    res=[] #makes empty list
    c=0 #c set to 0, when c > limit, the loop will end
    for record in fastq_parser: #starts loop that will iterate limit-number of times
        score=record.letter_annotations["phred_quality"] #SeqIO annotates the phred score for each bp of each read
        res.append(score) #appends the scores to thr list res
        c+=1 #add 1 to c. For each iteration
        if c>limit: #compares c to limit
            break #stops the loop if c > limit. If yes, the loop is terminated
    df = pd.DataFrame(res) #converts list res into a df
    l = len(df.T)+1 #makes l the length of df + 1 for check
    return df #returns df

df = getting_fastq_qualities(fastq_directory_R2, ax=None, limit=500000) #runs the above function

########For line plot on summary statistics for Phred scoores##############
summary = df.transpose() #transposes df
summary['position'] = pd.DataFrame(range(1,len(summary.columns) + 1)) #makes a columns named "positions" 
summary['average'] = summary.iloc[:,0:100].mean() #calulates average phred score for each position
summary['min'] = summary.iloc[:,0:100].min() #calulates the minimum phred score for each position
summary['max'] = summary.iloc[:,0:100].max() #calulates maximum phred score for each position
summary['std'] = np.std(summary.iloc[:,0:100]) #calulates standard deviation phred score for each position
summary #prits summary df

plt.plot('position', 'average', data=summary, markerfacecolor='Black', markersize=12, color='skyblue', linewidth=4)  #plots phred score "position" vs "average"
plt.plot('position', 'min', data=summary, marker='', color='blue', linewidth=2) #plots phred score "position" vs "minimum"
plt.plot('position', 'max', data=summary, marker='', color='red', linewidth=2, linestyle='dashed') #plots phred score "position" vs "maximum"
plt.plot('position', 'std', data=summary, marker='', color='olive', linewidth=2, linestyle='dashed') #plots phred score "position" vs "std"
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2) #moves legend box
plt.ylim((0,40)) #set a Y-axis limit
plt.xlabel("Position(bp)") #labels X-axis
plt.ylabel("Phred_Score") #labels Y-axis


#######For Phred Score heatmap############
def heatmap_fastq_qualities(filename, ax=None, limit=100): #Function used to make heatmap. Similar to the above function

    fastq_parser = SeqIO.parse(filename, "fastq") #opens the fastq file
    res=[] #makes empty list
    c=0 #c set to 0, when c > limit, the loop will end
    for record in fastq_parser: #starts loop that will iterate limit-number of times
        score=record.letter_annotations["phred_quality"] #SeqIO annotates the phred score for each bp of each read
        res.append(score) #appends the scores to thr list res
        c+=1 #add 1 to c. For each iteration
        if c>limit:  #compares c to limit
            break #stops the loop if c > limit. If yes, the loop is terminated
    df_heatmap = pd.DataFrame(res) #converts list res into a df
    l = len(df_heatmap.T)+1 #makes l the length of df + 1 for check
    return df_heatmap #returns df_heatmap

df_heatmap = heatmap_fastq_qualities(fastq_directory_R1, ax=None, limit=100) #runs function


fig, ax = plt.subplots(figsize=(30,30)) #creats a large subplot for the heatmap
sns.heatmap(df_heatmap,vmin=0, linewidths=0.3, linecolor="Black",center=0,annot=True,fmt="d") #makes the heatmap
plt.show() #shows heatmap



#######For "FastQC-like" quality figure###############
def visual_fastq_qualities(filename, ax=None, limit=800000): #Function used to make heatmap. Similar to the above function

    fastq_parser = SeqIO.parse(filename, "fastq") #opens the fastq file
    res=[] #makes empty list
    c=0 #c set to 0, when c > limit, the loop will end
    for record in fastq_parser: #starts loop that will iterate limit-number of times
        score=record.letter_annotations["phred_quality"] #SeqIO annotates the phred score for each bp of each read
        res.append(score) #appends the scores to thr list res
        c+=1 #add 1 to c. For each iteration
        if c>limit: #compares c to limit
            break #stops the loop if c > limit. If yes, the loop is terminated
    df = pd.DataFrame(res) #converts list res into a df
    l = len(df.T)+1 #makes l the length of df + 1 for check

    if ax==None:  #continues loop if ax == None. Defined in the function as "ax=None"
        f,ax=plt.subplots(figsize=(12,5))  #makes f and ax a subplot of a given size
    rect = patches.Rectangle((0,0),l,20,linewidth=0,facecolor='r',alpha=.4) #creates a shaded rectangle that is red and found throughout a defined range (bad phred scores)
    ax.add_patch(rect) #adds the rect patch to plot ax
    rect = patches.Rectangle((0,20),l,8,linewidth=0,facecolor='yellow',alpha=.4) #creates a shaded rectangle that is yellow and found throughout a defined range (moderate phred scores)
    ax.add_patch(rect) #adds the rect patch to plot ax
    rect = patches.Rectangle((0,28),l,12,linewidth=0,facecolor='g',alpha=.4) #creates a shaded rectangle that is green and found throughout a defined range (good phred scores)
    ax.add_patch(rect) #adds the rect patch to plot ax
    df.mean().plot(ax=ax,c='black') #plots the mean phred score as a black line for each bp
    boxprops = dict(linestyle='-', linewidth=1, color='black') #creates a dictionary
    df.plot(kind='box', ax=ax, grid=False, showfliers=False, #generates boxplots for each position
            color=dict(boxes='black',whiskers='black')  ) #defines the color of boxes and whiskers 
    ax.set_xticks(np.arange(0, l, 5)) #sets tick marks ith np.arange 
    ax.set_xticklabels(np.arange(0, l, 5)) #sets tick labels with np.arrang
    ax.set_xlabel('position(bp)') #labels the X-axis
    ax.set_xlim((0,l)) #sets X-axis limit to one bp at a time
    ax.set_ylim((0,40)) #sets Y-axis limit
    ax.set_title('per base sequence quality')  #labels the Y-axis   
    return #Returns plot

df = plot_fastq_qualities(fastq_directory_R2,limit=500000) #runs the above function