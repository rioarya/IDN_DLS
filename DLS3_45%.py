# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:21:55 2019

@author: raryapratama
"""

#%%

#Step (1): Import Python libraries, set land conversion scenarios general parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
import pandas as pd

#from matplotlib import rcParams


#rcParams.update({'figure.autolayout': False})


#DL_FP Scenario

##Set parameters
#Parameters for primary forest
initAGB = 233            #source: van Beijma et al. (2018)
initAGB_min = 233-72
initAGB_max = 233 + 72

#parameters for timber plantation. Source: Khasanah et al. (2015) 


tf = 201

a = 0.082
b = 2.53


#%%

#Step (2_1): C loss from the harvesting/clear cut


df2_Ac18 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')
df2_Tgr60 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')
dfE2_Hbr40 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')



t = range(0,tf,1)


c_firewood_energy_S2_Ac18 = df2_Ac18['Firewood_other_energy_use'].values
c_firewood_energy_S2_Tgr60 = df2_Tgr60['Firewood_other_energy_use'].values
c_firewood_energy_E2_Hbr40 = dfE2_Hbr40['Firewood_other_energy_use'].values



#%%

#Step (2_2): C loss from the harvesting/clear cut as wood pellets


dfE2 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')


c_pellets_Hbr_40y = dfE2['Wood_pellets'].values


#%%

#Step (3): Aboveground biomass (AGB) decomposition


#S2_Ac_18y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')

tf = 201

t = np.arange(tf)


def decomp_S2_Ac_18y(t,remainAGB_S2_Ac_18y):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_S2_Ac_18y



#set zero matrix
output_decomp_S2_Ac_18y = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_S2_Ac_18y in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_S2_Ac_18y[i:,i] = decomp_S2_Ac_18y(t[:len(t)-i],remain_part_S2_Ac_18y)

print(output_decomp_S2_Ac_18y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2_Ac_18y = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2_Ac_18y[:,i] = np.diff(output_decomp_S2_Ac_18y[:,i])
    i = i + 1 

print(subs_matrix_S2_Ac_18y[:,:4])
print(len(subs_matrix_S2_Ac_18y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2_Ac_18y = subs_matrix_S2_Ac_18y.clip(max=0)

print(subs_matrix_S2_Ac_18y[:,:4])

#make the results as absolute values
subs_matrix_S2_Ac_18y = abs(subs_matrix_S2_Ac_18y)
print(subs_matrix_S2_Ac_18y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2_Ac_18y = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_S2_Ac_18y)

subs_matrix_S2_Ac_18y = np.vstack((zero_matrix_S2_Ac_18y, subs_matrix_S2_Ac_18y))

print(subs_matrix_S2_Ac_18y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2_Ac_18y = (tf,1)
decomp_tot_S2_Ac_18y = np.zeros(matrix_tot_S2_Ac_18y) 

i = 0
while i < tf:
    decomp_tot_S2_Ac_18y[:,0] = decomp_tot_S2_Ac_18y[:,0] + subs_matrix_S2_Ac_18y[:,i]
    i = i + 1

print(decomp_tot_S2_Ac_18y[:,0])




#S2_Tgr_60y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')

tf = 201

t = np.arange(tf)


def decomp_S2_Tgr_60y(t,remainAGB_S2_Tgr_60y):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_S2_Tgr_60y



#set zero matrix
output_decomp_S2_Tgr_60y = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_S2_Tgr_60y in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_S2_Tgr_60y[i:,i] = decomp_S2_Tgr_60y(t[:len(t)-i],remain_part_S2_Tgr_60y)

print(output_decomp_S2_Tgr_60y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2_Tgr_60y = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2_Tgr_60y[:,i] = np.diff(output_decomp_S2_Tgr_60y[:,i])
    i = i + 1 

print(subs_matrix_S2_Tgr_60y[:,:4])
print(len(subs_matrix_S2_Tgr_60y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2_Tgr_60y = subs_matrix_S2_Tgr_60y.clip(max=0)

print(subs_matrix_S2_Tgr_60y[:,:4])

#make the results as absolute values
subs_matrix_S2_Tgr_60y = abs(subs_matrix_S2_Tgr_60y)
print(subs_matrix_S2_Tgr_60y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2_Tgr_60y = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_S2_Tgr_60y)

subs_matrix_S2_Tgr_60y = np.vstack((zero_matrix_S2_Tgr_60y, subs_matrix_S2_Tgr_60y))

print(subs_matrix_S2_Tgr_60y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2_Tgr_60y = (tf,1)
decomp_tot_S2_Tgr_60y = np.zeros(matrix_tot_S2_Tgr_60y) 

i = 0
while i < tf:
    decomp_tot_S2_Tgr_60y[:,0] = decomp_tot_S2_Tgr_60y[:,0] + subs_matrix_S2_Tgr_60y[:,i]
    i = i + 1

print(decomp_tot_S2_Tgr_60y[:,0])



#E
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')

tf = 201

t = np.arange(tf)


def decomp_E2_Hbr_40y(t,remainAGB_E2_Hbr_40y):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_E2_Hbr_40y



#set zero matrix
output_decomp_E2_Hbr_40y = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_E2_Hbr_40y in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_E2_Hbr_40y[i:,i] = decomp_E2_Hbr_40y(t[:len(t)-i],remain_part_E2_Hbr_40y)

print(output_decomp_E2_Hbr_40y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_E2_Hbr_40y = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_E2_Hbr_40y[:,i] = np.diff(output_decomp_E2_Hbr_40y[:,i])
    i = i + 1 

print(subs_matrix_E2_Hbr_40y[:,:4])
print(len(subs_matrix_E2_Hbr_40y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_E2_Hbr_40y = subs_matrix_E2_Hbr_40y.clip(max=0)

print(subs_matrix_E2_Hbr_40y[:,:4])

#make the results as absolute values
subs_matrix_E2_Hbr_40y = abs(subs_matrix_E2_Hbr_40y)
print(subs_matrix_E2_Hbr_40y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_E2_Hbr_40y = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_E2_Hbr_40y)

subs_matrix_E2_Hbr_40y = np.vstack((zero_matrix_E2_Hbr_40y, subs_matrix_E2_Hbr_40y))

print(subs_matrix_E2_Hbr_40y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_E2_Hbr_40y = (tf,1)
decomp_tot_E2_Hbr_40y = np.zeros(matrix_tot_E2_Hbr_40y) 

i = 0
while i < tf:
    decomp_tot_E2_Hbr_40y[:,0] = decomp_tot_E2_Hbr_40y[:,0] + subs_matrix_E2_Hbr_40y[:,i]
    i = i + 1

print(decomp_tot_E2_Hbr_40y[:,0])


#plotting
t = np.arange(0,tf)

plt.plot(t,decomp_tot_S2_Ac_18y,label='Ac_18y')
plt.plot(t,decomp_tot_S2_Tgr_60y,label='Tgr_60y')
plt.plot(t,decomp_tot_E2_Hbr_40y,label='E_Hbr_40y')

plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()



#%%

#Step (4): Dynamic stock model of in-use wood materials


from dynamic_stock_model import DynamicStockModel



df2_Ac18 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')
df2_Tgr60 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')
dfE2_Hbr40 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')


#product lifetime
#paper
P = 4

#furniture
F = 20

#building materials
B = 35


TestDSM2_Ac18 = DynamicStockModel(t = df2_Ac18['Year'].values, i = df2_Ac18['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([F]), 'StdDev': np.array([0.3*F])})
TestDSM2_Tgr60 = DynamicStockModel(t = df2_Tgr60['Year'].values, i = df2_Tgr60['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})
TestDSME2_Hbr40 = DynamicStockModel(t = dfE2_Hbr40['Year'].values, i = dfE2_Hbr40['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})



CheckStr2_Ac18, ExitFlag2_Ac18 = TestDSM2_Ac18.dimension_check()
CheckStr2_Tgr60, ExitFlag2_Tgr60 = TestDSM2_Tgr60.dimension_check()
CheckStrE2_Hbr40, ExitFlagE2_Hbr40 = TestDSME2_Hbr40.dimension_check()



Stock_by_cohort2_Ac18, ExitFlag2_Ac18 = TestDSM2_Ac18.compute_s_c_inflow_driven()
Stock_by_cohort2_Tgr60, ExitFlag2_Tgr60 = TestDSM2_Tgr60.compute_s_c_inflow_driven()
Stock_by_cohortE2_Hbr40, ExitFlagE2_Hbr40 = TestDSME2_Hbr40.compute_s_c_inflow_driven()




S2_Ac18, ExitFlag2_Ac18   = TestDSM2_Ac18.compute_stock_total()
S2_Tgr60, ExitFlag2_Tgr60   = TestDSM2_Tgr60.compute_stock_total()
S2_Hbr40, ExitFlagE2_Hbr40   = TestDSME2_Hbr40.compute_stock_total()



O_C2_Ac18, ExitFlag2_Ac18 = TestDSM2_Ac18.compute_o_c_from_s_c()
O_C2_Tgr60, ExitFlag2_Tgr60 = TestDSM2_Tgr60.compute_o_c_from_s_c()
O_CE2_Hbr40, ExitFlagE2_Hbr40 = TestDSME2_Hbr40.compute_o_c_from_s_c()




O2_Ac18, ExitFlag2_Ac18   = TestDSM2_Ac18.compute_outflow_total()
O2_Tgr60, ExitFlag2_Tgr60   = TestDSM2_Tgr60.compute_outflow_total()
OE2_Hbr40, ExitFlagE2_Hbr40   = TestDSME2_Hbr40.compute_outflow_total()





DS2_Ac18, ExitFlag2_Ac18  = TestDSM2_Ac18.compute_stock_change()
DS2_Tgr60, ExitFlag2_Tgr60  = TestDSM2_Tgr60.compute_stock_change()
DSE2_Hbr40, ExitFlagE2_Hbr40  = TestDSME2_Hbr40.compute_stock_change()



Bal2_Ac18, ExitFlag2_Ac18 = TestDSM2_Ac18.check_stock_balance()
Bal2_Tgr60, ExitFlag2_Tgr60 = TestDSM2_Tgr60.check_stock_balance()
BalE2_Hbr40, ExitFlagE2_Hbr40 = TestDSME2_Hbr40.check_stock_balance()



#print output flow

print(TestDSM2_Ac18.o)
print(TestDSM2_Tgr60.o)
print(TestDSME2_Hbr40.o)





#%%

#Step (5): Biomass growth


## one-year gap between rotation cycle

# A. crassicarpa (Source: Anitha et al., 2015; Adiriono, 2009). Code: Ac

tf_Ac_7y = 8
tf_Ac_18y = 19

A1 = range(1,tf_Ac_7y,1)
A2 = range(1,tf_Ac_18y,1)


#calculate the biomass and carbon content of A. crassicarpa over time (18y)
def Y_Ac_18y(A2):
    #return 44/12*1000*np.exp(4.503-(2.559/A2))
    #return 44/12*1000*1000000*np.exp(4.503-(2.559/A2))
    return 44/12*1000*789110.03296769*np.exp(4.503-(2.559/A2))
 

output_Y_Ac_18y = np.array([Y_Ac_18y(A2i) for A2i in A2])

print(output_Y_Ac_18y)

#insert 0 value to the first element of the output result
output_Y_Ac_18y = np.insert(output_Y_Ac_18y,0,0)

print(output_Y_Ac_18y)





##11 times 19-year cycle (+1 year gap after the FP harvest) of new AGB of A. crassicarpa (18y), zero year gap between the cycle
counter_18y = range(0,11,1)

y_Ac_18y = []

for i in counter_18y:
    y_Ac_18y.append(output_Y_Ac_18y)
    
flat_list_Ac_18y = []
for sublist in y_Ac_18y:
    for item in sublist:
        flat_list_Ac_18y.append(item)

        
        
#the length of the list is now 209, so we remove the last 8 elements of the list to make the len=tf
flat_list_Ac_18y = flat_list_Ac_18y[:len(flat_list_Ac_18y)-8]

  

#####Check the flat list length for Hbr


## T. grandis (Source: Anitha et al., 2015; Adiriono, 2009). Code: Tgr
tf_Tgr_60y = 61

T2 = range(0,tf_Tgr_60y,1)



#calculate the biomass and carbon content of T. grandis over time (60y)
def Y_Tgr_60y(T2):
    #return 44/12*1000*2.114*(T2**0.941)
    #return 44/12*1000*1000000*2.114*(T2**0.941)
    return 44/12*1000*706266.962012207*2.114*(T2**0.941)

output_Y_Tgr_60y = np.array([Y_Tgr_60y(T2i) for T2i in T2])

print(output_Y_Tgr_60y)




##4 times 60-year cycle of new AGB of T. grandis (60y), zero year gap between the cycle
counter_60y = range(0,4,1)

y_Tgr_60y = []

for i in counter_60y:
    y_Tgr_60y.append(output_Y_Tgr_60y)


    
flat_list_Tgr_60y = []
for sublist in y_Tgr_60y:
    for item in sublist:
        flat_list_Tgr_60y.append(item)

#the length of the list is now 244, so we remove the last 43 elements of the list to make the len=tf
flat_list_Tgr_60y = flat_list_Tgr_60y[:len(flat_list_Tgr_60y)-43]



## H. brasiliensis (Source: Guillaume et al., 2018). Code: Hbr

tf_Hbr_40y = 41

H1 = range(0,tf_Hbr_40y,1)


#calculate the biomass and carbon content of H. brasiliensis over time (40y)
def Y_Hbr_40y(H1):
    #return 44/12*1000*1.55*H1
    #return 44/12*1000*1.55*1000000*H1
    return 44/12*1000*1.55*442254.332634859*H1


output_Y_Hbr_40y = np.array([Y_Hbr_40y(H1i) for H1i in H1])

print(output_Y_Hbr_40y)


##5 times 40-year cycle of new AGB of H. brasiliensis (40y), zero year gap between the cycle
counter_40y = range(0,5,1)

y_Hbr_40y = []

for i in counter_40y:
    y_Hbr_40y.append(output_Y_Hbr_40y)


    
flat_list_Hbr_40y = []
for sublist in y_Hbr_40y:
    for item in sublist:
        flat_list_Hbr_40y.append(item)

#the length of the list is now 205, so we remove the last 4 elements of the list to make the len=tf
flat_list_Hbr_40y = flat_list_Hbr_40y[:len(flat_list_Hbr_40y)-4]


#plotting
t = range (0,tf,1)


plt.xlim([0, 200])


plt.plot(t, flat_list_Ac_18y, color='deeppink')
plt.plot(t, flat_list_Hbr_40y, color='darkviolet')
plt.plot(t, flat_list_Tgr_60y, color='seagreen')

#plt.fill_between(t, flat_list_nucleus, flat_list_plasma, color='darkseagreen', alpha='0.4')


plt.xlabel('Time (year)')
plt.ylabel('AGB (tC/ha)')

plt.show()





##Yearly sequestration


##A. crassicarpa (18y)
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_Ac_18y(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_Ac_18y = [t - u for u, t in zip(flat_list_Ac_18y, flat_list_Ac_18y[1:])]

#since there is no sequestration between the replanting year (e.g., year 25 to 26), we have to replace negative numbers in 'flat_list_Ac_18y' with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
flat_list_Ac_18y = [0 if i < 0 else i for i in flat_list_Ac_18y]

#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_Ac_18y.insert(0,var)

#make 'flat_list_plasma' elements negative numbers to denote sequestration
flat_list_Ac_18y = [ -x for x in flat_list_Ac_18y]


print(flat_list_Ac_18y)




##T. grandis (60y)
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_Tgr_60y(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_Tgr_60y = [k - l for l, k in zip(flat_list_Tgr_60y, flat_list_Tgr_60y[1:])]

#since there is no sequestration between the replanting year (e.g., year 25 to 26), we have to replace negative numbers in 'flat_list_Tgr_60y' with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
flat_list_Tgr_60y = [0 if i < 0 else i for i in flat_list_Tgr_60y]

#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_Tgr_60y.insert(0,var)

#make 'flat_list_plasma' elements negative numbers to denote sequestration
flat_list_Tgr_60y = [ -x for x in flat_list_Tgr_60y]


print(flat_list_Tgr_60y)



##H. brasiliensis (40y)
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_Hbr_40y(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_Hbr_40y = [c - d for d, c in zip(flat_list_Hbr_40y, flat_list_Hbr_40y[1:])]

#since there is no sequestration between the replanting year (e.g., year 25 to 26), we have to replace negative numbers in 'flat_list_Hbr_40y' with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
flat_list_Hbr_40y = [0 if i < 0 else i for i in flat_list_Hbr_40y]

#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_Hbr_40y.insert(0,var)

#make 'flat_list_plasma' elements negative numbers to denote sequestration
flat_list_Hbr_40y = [ -x for x in flat_list_Hbr_40y]


print(flat_list_Hbr_40y)


#%%

#Step (6): post-harvest processing of wood 


#post-harvest wood processing
df2_Ac_18y = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')
df2_Tgr_60y = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')
dfE2_Hbr_40y = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')



t = range(0,tf,1)

PH_Emissions_HWP2_Ac_18y = df2_Ac_18y['PH_Emissions_HWP'].values
PH_Emissions_HWP2_Tgr_60y = df2_Tgr_60y['PH_Emissions_HWP'].values
PH_Emissions_HWPE2_Hbr_40y = dfE2_Hbr_40y ['PH_Emissions_HWP'].values




#%%

#Step (7_1): landfill gas decomposition (CH4)

#CH4 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl




#S2_Ac_18y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')

tf = 201

t = np.arange(tf)


def decomp_CH4_S2_Ac_18y(t,remainAGB_CH4_S2_Ac_18y):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_S2_Ac_18y



#set zero matrix
output_decomp_CH4_S2_Ac_18y = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_S2_Ac_18y in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_S2_Ac_18y[i:,i] = decomp_CH4_S2_Ac_18y(t[:len(t)-i],remain_part_CH4_S2_Ac_18y)

print(output_decomp_CH4_S2_Ac_18y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_S2_Ac_18y = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_S2_Ac_18y[:,i] = np.diff(output_decomp_CH4_S2_Ac_18y[:,i])
    i = i + 1 

print(subs_matrix_CH4_S2_Ac_18y[:,:4])
print(len(subs_matrix_CH4_S2_Ac_18y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_S2_Ac_18y = subs_matrix_CH4_S2_Ac_18y.clip(max=0)

print(subs_matrix_CH4_S2_Ac_18y[:,:4])

#make the results as absolute values
subs_matrix_CH4_S2_Ac_18y = abs(subs_matrix_CH4_S2_Ac_18y)
print(subs_matrix_CH4_S2_Ac_18y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_S2_Ac_18y = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_S2_Ac_18y)

subs_matrix_CH4_S2_Ac_18y = np.vstack((zero_matrix_CH4_S2_Ac_18y, subs_matrix_CH4_S2_Ac_18y))

print(subs_matrix_CH4_S2_Ac_18y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_S2_Ac_18y = (tf,1)
decomp_tot_CH4_S2_Ac_18y = np.zeros(matrix_tot_CH4_S2_Ac_18y) 

i = 0
while i < tf:
    decomp_tot_CH4_S2_Ac_18y[:,0] = decomp_tot_CH4_S2_Ac_18y[:,0] + subs_matrix_CH4_S2_Ac_18y[:,i]
    i = i + 1

print(decomp_tot_CH4_S2_Ac_18y[:,0])


#S2_Tgr_60y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')

tf = 201

t = np.arange(tf)


def decomp_CH4_S2_Tgr_60y(t,remainAGB_CH4_S2_Tgr_60y):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_S2_Tgr_60y



#set zero matrix
output_decomp_CH4_S2_Tgr_60y = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_S2_Tgr_60y in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_S2_Tgr_60y[i:,i] = decomp_CH4_S2_Tgr_60y(t[:len(t)-i],remain_part_CH4_S2_Tgr_60y)

print(output_decomp_CH4_S2_Tgr_60y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_S2_Tgr_60y = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_S2_Tgr_60y[:,i] = np.diff(output_decomp_CH4_S2_Tgr_60y[:,i])
    i = i + 1 

print(subs_matrix_CH4_S2_Tgr_60y[:,:4])
print(len(subs_matrix_CH4_S2_Tgr_60y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_S2_Tgr_60y = subs_matrix_CH4_S2_Tgr_60y.clip(max=0)

print(subs_matrix_CH4_S2_Tgr_60y[:,:4])

#make the results as absolute values
subs_matrix_CH4_S2_Tgr_60y = abs(subs_matrix_CH4_S2_Tgr_60y)
print(subs_matrix_CH4_S2_Tgr_60y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_S2_Tgr_60y = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_S2_Tgr_60y)

subs_matrix_CH4_S2_Tgr_60y = np.vstack((zero_matrix_CH4_S2_Tgr_60y, subs_matrix_CH4_S2_Tgr_60y))

print(subs_matrix_CH4_S2_Tgr_60y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_S2_Tgr_60y = (tf,1)
decomp_tot_CH4_S2_Tgr_60y = np.zeros(matrix_tot_CH4_S2_Tgr_60y) 

i = 0
while i < tf:
    decomp_tot_CH4_S2_Tgr_60y[:,0] = decomp_tot_CH4_S2_Tgr_60y[:,0] + subs_matrix_CH4_S2_Tgr_60y[:,i]
    i = i + 1

print(decomp_tot_CH4_S2_Tgr_60y[:,0])



#E
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')

tf = 201

t = np.arange(tf)


def decomp_CH4_E2_Hbr_40y(t,remainAGB_CH4_E2_Hbr_40y):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_E2_Hbr_40y



#set zero matrix
output_decomp_CH4_E2_Hbr_40y = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_E2_Hbr_40y in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_E2_Hbr_40y[i:,i] = decomp_CH4_E2_Hbr_40y(t[:len(t)-i],remain_part_CH4_E2_Hbr_40y)

print(output_decomp_CH4_E2_Hbr_40y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_E2_Hbr_40y = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_E2_Hbr_40y[:,i] = np.diff(output_decomp_CH4_E2_Hbr_40y[:,i])
    i = i + 1 

print(subs_matrix_CH4_E2_Hbr_40y[:,:4])
print(len(subs_matrix_CH4_E2_Hbr_40y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_E2_Hbr_40y = subs_matrix_CH4_E2_Hbr_40y.clip(max=0)

print(subs_matrix_CH4_E2_Hbr_40y[:,:4])

#make the results as absolute values
subs_matrix_CH4_E2_Hbr_40y = abs(subs_matrix_CH4_E2_Hbr_40y)
print(subs_matrix_CH4_E2_Hbr_40y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_E2_Hbr_40y = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_E2_Hbr_40y)

subs_matrix_CH4_E2_Hbr_40y = np.vstack((zero_matrix_CH4_E2_Hbr_40y, subs_matrix_CH4_E2_Hbr_40y))

print(subs_matrix_CH4_E2_Hbr_40y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_E2_Hbr_40y = (tf,1)
decomp_tot_CH4_E2_Hbr_40y = np.zeros(matrix_tot_CH4_E2_Hbr_40y) 

i = 0
while i < tf:
    decomp_tot_CH4_E2_Hbr_40y[:,0] = decomp_tot_CH4_E2_Hbr_40y[:,0] + subs_matrix_CH4_E2_Hbr_40y[:,i]
    i = i + 1

print(decomp_tot_CH4_E2_Hbr_40y[:,0])


#plotting
t = np.arange(0,tf)


plt.plot(t,decomp_tot_CH4_S2_Ac_18y,label='Ac_18y')
plt.plot(t,decomp_tot_CH4_S2_Tgr_60y,label='Tgr_60y')
plt.plot(t,decomp_tot_CH4_E2_Hbr_40y,label='E_Hbr_40y')

plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()



#%%

#Step (7_2): landfill gas decomposition (CO2)

#CO2 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl

#Step (2) Aboveground biomass (AGB) decomposition



#S2_Ac_18y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')

tf = 201

t = np.arange(tf)


def decomp_S2_Ac_18y(t,remainAGB_S2_Ac_18y):
    return (1-(1-np.exp(-k*t)))*remainAGB_S2_Ac_18y



#set zero matrix
output_decomp_S2_Ac_18y = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_S2_Ac_18y in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_S2_Ac_18y[i:,i] = decomp_S2_Ac_18y(t[:len(t)-i],remain_part_S2_Ac_18y)

print(output_decomp_S2_Ac_18y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2_Ac_18y = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2_Ac_18y[:,i] = np.diff(output_decomp_S2_Ac_18y[:,i])
    i = i + 1 

print(subs_matrix_S2_Ac_18y[:,:4])
print(len(subs_matrix_S2_Ac_18y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2_Ac_18y = subs_matrix_S2_Ac_18y.clip(max=0)

print(subs_matrix_S2_Ac_18y[:,:4])

#make the results as absolute values
subs_matrix_S2_Ac_18y = abs(subs_matrix_S2_Ac_18y)
print(subs_matrix_S2_Ac_18y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2_Ac_18y = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_S2_Ac_18y)

subs_matrix_S2_Ac_18y = np.vstack((zero_matrix_S2_Ac_18y, subs_matrix_S2_Ac_18y))

print(subs_matrix_S2_Ac_18y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2_Ac_18y = (tf,1)
decomp_tot_CO2_S2_Ac_18y = np.zeros(matrix_tot_S2_Ac_18y) 

i = 0
while i < tf:
    decomp_tot_CO2_S2_Ac_18y[:,0] = decomp_tot_CO2_S2_Ac_18y[:,0] + subs_matrix_S2_Ac_18y[:,i]
    i = i + 1

print(decomp_tot_CO2_S2_Ac_18y[:,0])




#S2_Tgr_60y
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')

tf = 201

t = np.arange(tf)


def decomp_S2_Tgr_60y(t,remainAGB_S2_Tgr_60y):
    return (1-(1-np.exp(-k*t)))*remainAGB_S2_Tgr_60y



#set zero matrix
output_decomp_S2_Tgr_60y = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_S2_Tgr_60y in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_S2_Tgr_60y[i:,i] = decomp_S2_Tgr_60y(t[:len(t)-i],remain_part_S2_Tgr_60y)

print(output_decomp_S2_Tgr_60y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2_Tgr_60y = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2_Tgr_60y[:,i] = np.diff(output_decomp_S2_Tgr_60y[:,i])
    i = i + 1 

print(subs_matrix_S2_Tgr_60y[:,:4])
print(len(subs_matrix_S2_Tgr_60y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2_Tgr_60y = subs_matrix_S2_Tgr_60y.clip(max=0)

print(subs_matrix_S2_Tgr_60y[:,:4])

#make the results as absolute values
subs_matrix_S2_Tgr_60y = abs(subs_matrix_S2_Tgr_60y)
print(subs_matrix_S2_Tgr_60y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2_Tgr_60y = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_S2_Tgr_60y)

subs_matrix_S2_Tgr_60y = np.vstack((zero_matrix_S2_Tgr_60y, subs_matrix_S2_Tgr_60y))

print(subs_matrix_S2_Tgr_60y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2_Tgr_60y = (tf,1)
decomp_tot_CO2_S2_Tgr_60y = np.zeros(matrix_tot_S2_Tgr_60y) 

i = 0
while i < tf:
    decomp_tot_CO2_S2_Tgr_60y[:,0] = decomp_tot_CO2_S2_Tgr_60y[:,0] + subs_matrix_S2_Tgr_60y[:,i]
    i = i + 1

print(decomp_tot_CO2_S2_Tgr_60y[:,0])



#E
df = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')

tf = 201

t = np.arange(tf)


def decomp_E2_Hbr_40y(t,remainAGB_E2_Hbr_40y):
    return (1-(1-np.exp(-k*t)))*remainAGB_E2_Hbr_40y



#set zero matrix
output_decomp_E2_Hbr_40y = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_E2_Hbr_40y in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_E2_Hbr_40y[i:,i] = decomp_E2_Hbr_40y(t[:len(t)-i],remain_part_E2_Hbr_40y)

print(output_decomp_E2_Hbr_40y[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_E2_Hbr_40y = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_E2_Hbr_40y[:,i] = np.diff(output_decomp_E2_Hbr_40y[:,i])
    i = i + 1 

print(subs_matrix_E2_Hbr_40y[:,:4])
print(len(subs_matrix_E2_Hbr_40y))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_E2_Hbr_40y = subs_matrix_E2_Hbr_40y.clip(max=0)

print(subs_matrix_E2_Hbr_40y[:,:4])

#make the results as absolute values
subs_matrix_E2_Hbr_40y = abs(subs_matrix_E2_Hbr_40y)
print(subs_matrix_E2_Hbr_40y[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_E2_Hbr_40y = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_E2_Hbr_40y)

subs_matrix_E2_Hbr_40y = np.vstack((zero_matrix_E2_Hbr_40y, subs_matrix_E2_Hbr_40y))

print(subs_matrix_E2_Hbr_40y[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_E2_Hbr_40y = (tf,1)
decomp_tot_CO2_E2_Hbr_40y = np.zeros(matrix_tot_E2_Hbr_40y) 

i = 0
while i < tf:
    decomp_tot_CO2_E2_Hbr_40y[:,0] = decomp_tot_CO2_E2_Hbr_40y[:,0] + subs_matrix_E2_Hbr_40y[:,i]
    i = i + 1

print(decomp_tot_CO2_E2_Hbr_40y[:,0])


#plotting
t = np.arange(0,tf)


plt.plot(t,decomp_tot_CO2_S2_Ac_18y,label='Ac_18y')
plt.plot(t,decomp_tot_CO2_S2_Tgr_60y,label='Tgr_60y')
plt.plot(t,decomp_tot_CO2_E2_Hbr_40y,label='E_Hbr_40y')

plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()



#%%

#Step (8): Sum the emissions and sequestration (net carbon balance), CO2 and CH4 are separated


#https://stackoverflow.com/questions/52703442/python-sum-values-from-multiple-lists-more-than-two
#C_loss + C_remainAGB + C_remainHWP + PH_Emissions_PO


Emissions_S2_Ac_18y = [c_firewood_energy_S2_Ac18, decomp_tot_S2_Ac_18y[:,0], TestDSM2_Ac18.o, PH_Emissions_HWP2_Ac_18y, decomp_tot_CO2_S2_Ac_18y[:,0]]
Emissions_S2_Tgr_60y = [c_firewood_energy_S2_Tgr60, decomp_tot_S2_Tgr_60y[:,0], TestDSM2_Tgr60.o, PH_Emissions_HWP2_Tgr_60y, decomp_tot_CO2_S2_Tgr_60y[:,0]]
Emissions_E2_Hbr_40y = [c_firewood_energy_E2_Hbr40, c_pellets_Hbr_40y, decomp_tot_E2_Hbr_40y[:,0], TestDSME2_Hbr40.o, PH_Emissions_HWPE2_Hbr_40y, decomp_tot_CO2_E2_Hbr_40y[:,0]]



Emissions_DL_FP_S2_Ac_18y = [sum(x) for x in zip(*Emissions_S2_Ac_18y)]
Emissions_DL_FP_S2_Tgr_60y = [sum(x) for x in zip(*Emissions_S2_Tgr_60y)]
Emissions_DL_FP_E2_Hbr_40y = [sum(x) for x in zip(*Emissions_E2_Hbr_40y)]



#CH4_S2_Ac_18y
Emissions_CH4_DL_FP_S2_Ac_18y = decomp_tot_CH4_S2_Ac_18y[:,0]


#CH4_S2_Tgr_60y
Emissions_CH4_DL_FP_S2_Tgr_60y = decomp_tot_CH4_S2_Tgr_60y[:,0]

#CH4_E2_Hbr_40y
Emissions_CH4_DL_FP_E2_Hbr_40y = decomp_tot_CH4_E2_Hbr_40y[:,0]



#%%

#Step (9): Generate the excel file (emissions_seq_scenarios.xlsx) from Step (8) calculation


#print year column
year = []
for x in range (0, tf):
    year.append(x)
print (year)



#print CH4 emission column
import itertools
lst = [0]
Emissions_CH4 = list(itertools.chain.from_iterable(itertools.repeat(x, tf) for x in lst))
print(Emissions_CH4)


#print emission ref 
lst1 = [0]
Emission_ref = list(itertools.chain.from_iterable(itertools.repeat(x, tf) for x in lst1))
print(Emission_ref)



#replace the first element with 1 to denote the emission reference as year 0 (for dynGWP calculation)
Emission_ref[0] = 1
print(Emission_ref)





Col1 = year

Col2_S2_Ac_18y = Emissions_DL_FP_S2_Ac_18y
Col2_S2_Tgr_60y = Emissions_DL_FP_S2_Tgr_60y
Col2_E2_Hbr_40y = Emissions_DL_FP_E2_Hbr_40y


Col3_S2_Ac_18y = Emissions_CH4_DL_FP_S2_Ac_18y
Col3_S2_Tgr_60y = Emissions_CH4_DL_FP_S2_Tgr_60y
Col3_E2_Hbr_40y = Emissions_CH4_DL_FP_E2_Hbr_40y


Col4 = Emission_ref
Col6 = flat_list_Ac_18y
Col8 = flat_list_Tgr_60y
Col9 = flat_list_Hbr_40y

#A. crassicarpa

df2_Ac_18y = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2_Ac_18y,'kg_CH4':Col3_S2_Ac_18y,'kg_CO2_seq':Col6,'emission_ref':Col4})

#T. grandis
df2_Tgr_60y = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2_Tgr_60y,'kg_CH4':Col3_S2_Tgr_60y,'kg_CO2_seq':Col8,'emission_ref':Col4})

#H. brasiliensis
dfE2_Hbr_40y = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E2_Hbr_40y,'kg_CH4':Col3_E2_Hbr_40y,'kg_CO2_seq':Col9,'emission_ref':Col4})


writer = pd.ExcelWriter('emissions_seq_DLS3.xlsx', engine = 'xlsxwriter')


df2_Ac_18y.to_excel(writer, sheet_name = 'DLS3_45%_Ac_18y', header=True, index=False)
df2_Tgr_60y.to_excel(writer, sheet_name = 'DLS3_45%_Tgr', header=True, index=False)
dfE2_Hbr_40y.to_excel(writer, sheet_name = 'DLS3_60%_Hbr', header=True, index=False)


writer.save()





#%%

## DYNAMIC LCA - wood-based scenarios

# Step (10): Set General Parameters for Dynamic LCA calculation

aCH4 = 0.129957e-12;    # methane - instantaneous radiative forcing per unit mass [W/m2 /kgCH4]
TauCH4 = 12;    # methane - lifetime (years)
aCO2 = 0.0018088e-12;    # CO2 - instantaneous radiative forcing per unit mass [W/m2 /kgCO2]
TauCO2 = [172.9,  18.51,  1.186];    # CO2 parameters according to Bern carbon cycle-climate model
aBern = [0.259, 0.338, 0.186];        # CO2 parameters according to Bern carbon cycle-climate model
a0Bern = 0.217;                     # CO2 parameters according to Bern carbon cycle-climate model
tf = 202                           #until 202 because we want to get the DCF(t-i) until DCF(201) to determine the impact from the emission from the year 200 (There is no DCF(0))


#%%

#Step (11): Bern 2.5 CC Model, determine atmospheric load (C(t)) for GHG (CO2 and CH4)

t = range(0,tf,1)


## CO2 calculation formula
# time dependant atmospheric load for CO2, Bern model
def C_CO2(t):
    return a0Bern + aBern[0]*np.exp(-t/TauCO2[0]) + aBern[1]*np.exp(-t/TauCO2[1]) + aBern[2]*np.exp(-t/TauCO2[2])

output_CO2 = np.array([C_CO2(ti) for ti in t])

print(output_CO2)


## CH4 calculation formula
# time dependant atmospheric load for non-CO2 GHGs (Methane)
def C_CH4(t):
    return np.exp(-t/TauCH4)

output_CH4 = np.array([C_CH4(ti) for ti in t])




plt.xlim([0, 200])
plt.ylim([0,1.1])

plt.plot(t, output_CO2, output_CH4)


plt.xlabel('Time (year)')
plt.ylabel('Fraction of CO$_2$')

plt.show()


output_CH4.size
#%%

#determine the C(t) for CO2
s = []

t = np.arange(0,tf,1)

for i in t:
    s.append(quad(C_CO2,i-1,i))
    
res_list_CO2 = [x[0] for x in s]

len(res_list_CO2)

#%%

#determine the C(t) for CH4
s = []

for i in t:
    s.append(quad(C_CH4,i-1,i))

res_list_CH4 = [p[0] for p in s]


#plot
plt.xlim([0, 200])
plt.ylim([0,1.5])

 
plt.plot(t, res_list_CO2, res_list_CH4)
plt.show()

#%%

#Step (12): Determine dynamic characterization factors (DCF) for CO2 and CH4

DCF_inst_CO2 = aCO2 * np.array(res_list_CO2)


print(DCF_inst_CO2)


DCF_inst_CH4 = aCH4 * np.array(res_list_CH4)


plt.xlim([0, 200])
plt.ylim([0,4e-15])


plt.plot(t, DCF_inst_CO2, DCF_inst_CH4)
plt.xlabel('Time (year)')
plt.ylabel('DCF_inst (10$^{-15}$ W/m$^2$.kg CO$_2$)')
plt.show()

len(DCF_inst_CO2)

#%%

#Step (13): import emission data from emissions_seq_scenarios.xlsx (Step (9))

##wood-based


#read S2_Ac_18y
df = pd.read_excel('emissions_seq_DLS3.xlsx', 'DLS3_45%_Ac_18y')
emission_CO2_S2_Ac_18y = df['kg_CO2'].tolist()
emission_CH4_S2_Ac_18y = df['kg_CH4'].tolist()
emission_CO2_seq_S2_Ac_18y = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 


#read S2_Tgr_60y
df = pd.read_excel('emissions_seq_DLS3.xlsx', 'DLS3_45%_Tgr')
emission_CO2_S2_Tgr_60y = df['kg_CO2'].tolist()
emission_CH4_S2_Tgr_60y = df['kg_CH4'].tolist()
emission_CO2_seq_S2_Tgr_60y = df['kg_CO2_seq'].tolist()

#read E_Hbr_40y
df = pd.read_excel('emissions_seq_DLS3.xlsx', 'DLS3_60%_Hbr') # can also index sheet by name or fetch all sheets
emission_CO2_E2_Hbr_40y = df['kg_CO2'].tolist()
emission_CH4_E2_Hbr_40y = df['kg_CH4'].tolist()
emission_CO2_seq_E2_Hbr_40y = df['kg_CO2_seq'].tolist()


#%%

#Step (14a): import emission data from the counter-use of non-renewable materials/energy scenarios (NR)


emission_CO2_ref = df['emission_ref'].tolist()


#read S2_Ac_18y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_45%_Ac_18y')
emissions_NonRW_S2_Ac_18y = df['NonRW_emissions'].tolist()
emissions_NonRW_S2_Ac_18y_seq = df['kg_CO2_seq'].tolist()




#read S2_Tgr_60y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_45%_Tgr')
emissions_NonRW_S2_Tgr_60y = df['NonRW_emissions'].tolist()
emissions_NonRW_S2_Tgr_60y_seq = df['kg_CO2_seq'].tolist()


#read E2_Hbr_40y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_60%_Hbr') # can also index sheet by name or fetch all sheets
emissions_NonRW_E2_Hbr_40y = df['NonRW_emissions'].tolist()
emissions_NonRW_E2_Hbr_40y_seq = df['kg_CO2_seq'].tolist()

#%%

#Step (14b): import emission data from the reforestation scenarios (Refor)

#read S2_Ac_18y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_45%_Ac_18y')
emissions_refor_S2_Ac_18y = df['NonRW_emissions'].tolist()
emissions_refor_S2_Ac_18y_seq = df['kg_CO2_seq_refor'].tolist()



#read S2_Tgr_60y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_45%_Tgr')
emissions_refor_S2_Tgr_60y = df['NonRW_emissions'].tolist()
emissions_refor_S2_Tgr_60y_seq = df['kg_CO2_seq_refor'].tolist()


#read E2_Hbr_40y
df = pd.read_excel('NonRW_DLS3.xlsx', 'DLS3_60%_Hbr') # can also index sheet by name or fetch all sheets
emissions_refor_E2_Hbr_40y = df['NonRW_emissions'].tolist()
emissions_refor_E2_Hbr_40y_seq = df['kg_CO2_seq_refor'].tolist()



#%%

#Step (15): Determine the time elapsed dynamic characterization factors, DCF(t-ti), for CO2 and CH4

#DCF(t-i) CO2
matrix = (tf-1,tf-1)
DCF_CO2_ti = np.zeros(matrix)

for t in range(0,tf-1):
    i = -1
    while i < t:
        DCF_CO2_ti[i+1,t] = DCF_inst_CO2[t-i]
        i = i + 1

print(DCF_CO2_ti)

#sns.heatmap(DCF_CO2_ti)

DCF_CO2_ti.shape



#DCF(t-i) CH4
matrix = (tf-1,tf-1)
DCF_CH4_ti = np.zeros(matrix)

for t in range(0,tf-1):
    i = -1
    while i < t:
        DCF_CH4_ti[i+1,t] = DCF_inst_CH4[t-i]
        i = i + 1

print(DCF_CH4_ti)
#sns.heatmap(DCF_CH4_ti)

DCF_CH4_ti.shape

#%%

# Step (16): Calculate instantaneous global warming impact (GWI) 


##Wood-based

#S2_Ac_18y
t = np.arange(0,tf-1,1)

matrix_GWI_S2_Ac_18y = (tf-1,3)
GWI_inst_S2_Ac_18y = np.zeros(matrix_GWI_S2_Ac_18y)



for t in range(0,tf-1):
    GWI_inst_S2_Ac_18y[t,0] = np.sum(np.multiply(emission_CO2_S2_Ac_18y,DCF_CO2_ti[:,t]))
    GWI_inst_S2_Ac_18y[t,1] = np.sum(np.multiply(emission_CH4_S2_Ac_18y,DCF_CH4_ti[:,t]))
    GWI_inst_S2_Ac_18y[t,2] = np.sum(np.multiply(emission_CO2_seq_S2_Ac_18y,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2_Ac_18y = (tf-1,1)
GWI_inst_tot_S2_Ac_18y = np.zeros(matrix_GWI_tot_S2_Ac_18y)

GWI_inst_tot_S2_Ac_18y[:,0] = np.array(GWI_inst_S2_Ac_18y[:,0] + GWI_inst_S2_Ac_18y[:,1] + GWI_inst_S2_Ac_18y[:,2])
  
print(GWI_inst_tot_S2_Ac_18y[:,0])


#S2_Tgr_60y
t = np.arange(0,tf-1,1)

matrix_GWI_S2_Tgr_60y = (tf-1,3)
GWI_inst_S2_Tgr_60y = np.zeros(matrix_GWI_S2_Tgr_60y)



for t in range(0,tf-1):
    GWI_inst_S2_Tgr_60y[t,0] = np.sum(np.multiply(emission_CO2_S2_Tgr_60y,DCF_CO2_ti[:,t]))
    GWI_inst_S2_Tgr_60y[t,1] = np.sum(np.multiply(emission_CH4_S2_Tgr_60y,DCF_CH4_ti[:,t]))
    GWI_inst_S2_Tgr_60y[t,2] = np.sum(np.multiply(emission_CO2_seq_S2_Tgr_60y,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2_Tgr_60y = (tf-1,1)
GWI_inst_tot_S2_Tgr_60y = np.zeros(matrix_GWI_tot_S2_Tgr_60y)

GWI_inst_tot_S2_Tgr_60y[:,0] = np.array(GWI_inst_S2_Tgr_60y[:,0] + GWI_inst_S2_Tgr_60y[:,1] + GWI_inst_S2_Tgr_60y[:,2])
  
print(GWI_inst_tot_S2_Tgr_60y[:,0])


#E_Hbr_40y
t = np.arange(0,tf-1,1)

matrix_GWI_E2_Hbr_40y = (tf-1,3)
GWI_inst_E2_Hbr_40y = np.zeros(matrix_GWI_E2_Hbr_40y)



for t in range(0,tf-1):
    GWI_inst_E2_Hbr_40y[t,0] = np.sum(np.multiply(emission_CO2_E2_Hbr_40y,DCF_CO2_ti[:,t]))
    GWI_inst_E2_Hbr_40y[t,1] = np.sum(np.multiply(emission_CH4_E2_Hbr_40y,DCF_CH4_ti[:,t]))
    GWI_inst_E2_Hbr_40y[t,2] = np.sum(np.multiply(emission_CO2_seq_E2_Hbr_40y,DCF_CO2_ti[:,t]))

matrix_GWI_tot_E2_Hbr_40y = (tf-1,1)
GWI_inst_tot_E2_Hbr_40y = np.zeros(matrix_GWI_tot_E2_Hbr_40y)

GWI_inst_tot_E2_Hbr_40y[:,0] = np.array(GWI_inst_E2_Hbr_40y[:,0] + GWI_inst_E2_Hbr_40y[:,1] + GWI_inst_E2_Hbr_40y[:,2])
  
print(GWI_inst_tot_E2_Hbr_40y[:,0])

#%%
##NonRW

#S2_Ac_18y
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_S2_Ac_18y = (tf-1,2)
GWI_inst_NonRW_S2_Ac_18y = np.zeros(matrix_GWI_NonRW_S2_Ac_18y)



for t in range(0,tf-1):
    GWI_inst_NonRW_S2_Ac_18y[t,0] = np.sum(np.multiply(emissions_NonRW_S2_Ac_18y,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2_Ac_18y[t,1] = np.sum(np.multiply(emissions_NonRW_S2_Ac_18y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_NonRW_S2_Ac_18y = (tf-1,1)
GWI_inst_tot_NonRW_S2_Ac_18y = np.zeros(matrix_GWI_tot_NonRW_S2_Ac_18y)

GWI_inst_tot_NonRW_S2_Ac_18y[:,0] = np.array(GWI_inst_NonRW_S2_Ac_18y[:,0] + GWI_inst_NonRW_S2_Ac_18y[:,1])
print(GWI_inst_tot_NonRW_S2_Ac_18y[:,0])




#S2_Tgr_60y
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_S2_Tgr_60y = (tf-1,2)
GWI_inst_NonRW_S2_Tgr_60y = np.zeros(matrix_GWI_NonRW_S2_Tgr_60y)



for t in range(0,tf-1):
    GWI_inst_NonRW_S2_Tgr_60y[t,0] = np.sum(np.multiply(emissions_NonRW_S2_Tgr_60y,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2_Tgr_60y[t,1] = np.sum(np.multiply(emissions_NonRW_S2_Tgr_60y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_NonRW_S2_Tgr_60y = (tf-1,1)
GWI_inst_tot_NonRW_S2_Tgr_60y = np.zeros(matrix_GWI_tot_NonRW_S2_Tgr_60y)

GWI_inst_tot_NonRW_S2_Tgr_60y[:,0] = np.array(GWI_inst_NonRW_S2_Tgr_60y[:,0] + GWI_inst_NonRW_S2_Tgr_60y[:,1])
  
print(GWI_inst_tot_NonRW_S2_Tgr_60y[:,0])


#E2_Hbr_40y
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_E2_Hbr_40y = (tf-1,2)
GWI_inst_NonRW_E2_Hbr_40y = np.zeros(matrix_GWI_NonRW_E2_Hbr_40y)



for t in range(0,tf-1):
    GWI_inst_NonRW_E2_Hbr_40y[t,0] = np.sum(np.multiply(emissions_NonRW_E2_Hbr_40y,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_E2_Hbr_40y[t,1] = np.sum(np.multiply(emissions_NonRW_E2_Hbr_40y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_NonRW_E2_Hbr_40y = (tf-1,1)
GWI_inst_tot_NonRW_E2_Hbr_40y = np.zeros(matrix_GWI_tot_NonRW_E2_Hbr_40y)

GWI_inst_tot_NonRW_E2_Hbr_40y[:,0] = np.array(GWI_inst_NonRW_E2_Hbr_40y[:,0] + GWI_inst_NonRW_E2_Hbr_40y[:,1])
  
print(GWI_inst_tot_NonRW_E2_Hbr_40y[:,0])


#%% Reforestation


#S2_Ac_18y
t = np.arange(0,tf-1,1)

matrix_GWI_refor_S2_Ac_18y = (tf-1,2)
GWI_inst_refor_S2_Ac_18y = np.zeros(matrix_GWI_refor_S2_Ac_18y)



for t in range(0,tf-1):
    GWI_inst_refor_S2_Ac_18y[t,0] = np.sum(np.multiply(emissions_refor_S2_Ac_18y,DCF_CO2_ti[:,t]))
    GWI_inst_refor_S2_Ac_18y[t,1] = np.sum(np.multiply(emissions_refor_S2_Ac_18y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_refor_S2_Ac_18y = (tf-1,1)
GWI_inst_tot_refor_S2_Ac_18y = np.zeros(matrix_GWI_tot_refor_S2_Ac_18y)

GWI_inst_tot_refor_S2_Ac_18y[:,0] = np.array(GWI_inst_refor_S2_Ac_18y[:,0] + GWI_inst_refor_S2_Ac_18y[:,1])
print(GWI_inst_tot_refor_S2_Ac_18y[:,0])




#S2_Tgr_60y
t = np.arange(0,tf-1,1)

matrix_GWI_refor_S2_Tgr_60y = (tf-1,2)
GWI_inst_refor_S2_Tgr_60y = np.zeros(matrix_GWI_refor_S2_Tgr_60y)



for t in range(0,tf-1):
    GWI_inst_refor_S2_Tgr_60y[t,0] = np.sum(np.multiply(emissions_refor_S2_Tgr_60y,DCF_CO2_ti[:,t]))
    GWI_inst_refor_S2_Tgr_60y[t,1] = np.sum(np.multiply(emissions_refor_S2_Tgr_60y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_refor_S2_Tgr_60y = (tf-1,1)
GWI_inst_tot_refor_S2_Tgr_60y = np.zeros(matrix_GWI_tot_refor_S2_Tgr_60y)

GWI_inst_tot_refor_S2_Tgr_60y[:,0] = np.array(GWI_inst_refor_S2_Tgr_60y[:,0] + GWI_inst_refor_S2_Tgr_60y[:,1])
  
print(GWI_inst_tot_refor_S2_Tgr_60y[:,0])


#E2_Hbr_40y
t = np.arange(0,tf-1,1)

matrix_GWI_refor_E2_Hbr_40y = (tf-1,2)
GWI_inst_refor_E2_Hbr_40y = np.zeros(matrix_GWI_refor_E2_Hbr_40y)



for t in range(0,tf-1):
    GWI_inst_refor_E2_Hbr_40y[t,0] = np.sum(np.multiply(emissions_refor_E2_Hbr_40y,DCF_CO2_ti[:,t]))
    GWI_inst_refor_E2_Hbr_40y[t,1] = np.sum(np.multiply(emissions_refor_E2_Hbr_40y_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_refor_E2_Hbr_40y = (tf-1,1)
GWI_inst_tot_refor_E2_Hbr_40y = np.zeros(matrix_GWI_tot_refor_E2_Hbr_40y)

GWI_inst_tot_refor_E2_Hbr_40y[:,0] = np.array(GWI_inst_refor_E2_Hbr_40y[:,0] + GWI_inst_refor_E2_Hbr_40y[:,1])
  
print(GWI_inst_tot_refor_E2_Hbr_40y[:,0])


#%% Instantaneous GWI


t = np.arange(0,tf-1,1)

#create zero list to highlight the horizontal line for 0
def zerolistmaker(n):
    listofzeros = [0] * (n)
    return listofzeros

#convert to flat list
GWI_inst_tot_NonRW_S2_Ac_18y = np.array([item for sublist in GWI_inst_tot_NonRW_S2_Ac_18y for item in sublist])
GWI_inst_tot_NonRW_S2_Tgr_60y = np.array([item for sublist in GWI_inst_tot_NonRW_S2_Tgr_60y for item in sublist])
GWI_inst_tot_NonRW_E2_Hbr_40y = np.array([item for sublist in GWI_inst_tot_NonRW_E2_Hbr_40y for item in sublist])


GWI_inst_tot_refor_S2_Ac_18y = np.array([item for sublist in GWI_inst_tot_refor_S2_Ac_18y for item in sublist])
GWI_inst_tot_refor_S2_Tgr_60y = np.array([item for sublist in GWI_inst_tot_refor_S2_Tgr_60y for item in sublist])
GWI_inst_tot_refor_E2_Hbr_40y = np.array([item for sublist in GWI_inst_tot_refor_E2_Hbr_40y for item in sublist])


GWI_inst_tot_S2_Ac_18y = np.array([item for sublist in GWI_inst_tot_S2_Ac_18y for item in sublist])
GWI_inst_tot_S2_Tgr_60y = np.array([item for sublist in GWI_inst_tot_S2_Tgr_60y for item in sublist])
GWI_inst_tot_E2_Hbr_40y = np.array([item for sublist in GWI_inst_tot_E2_Hbr_40y for item in sublist])


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)



ax.plot(t, GWI_inst_tot_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax.plot(t, GWI_inst_tot_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax.plot(t, GWI_inst_tot_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax.plot(t, GWI_inst_tot_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax.plot(t, GWI_inst_tot_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax.plot(t, GWI_inst_tot_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax.plot(t, GWI_inst_tot_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax.plot(t, GWI_inst_tot_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax.plot(t, GWI_inst_tot_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_inst_tot_NonRW_E2_Hbr_40y, GWI_inst_tot_NonRW_S2_Tgr_60y, color='lightcoral', alpha=0.3)
#plt.fill_between(t, GWI_inst_tot_NonRW_S2_Ac_7y, GWI_inst_tot_NonRW_S2_Tgr_60y, color='lightcoral', alpha=0.3)




ax.set_xlim(0,200)
#plt.ylim(-2.5e-9,2e-9)
#ax.set_ylim(-0.6e-9,1.5e-9)   
#ax.set_ylim(-0.001,0.003)  
ax.set_ylim(-0.001,0.002) 

#from matplotlib.ticker import FormatStrFormatter

#ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.yticks(np.arange(-0.001, 0.0021, 0.001))

ax.set_title('Instantaneous GWI, DLS3', fontsize=14)

ax.set_xlabel('Time (year)', fontsize=14)
#plt.ylabel('GWI_inst (10$^{-12}$ W/m$^2$)')
ax.set_ylabel('GWI_inst (W/(m$^2$.year))', fontsize=14)

#major_yticks = np.arange(-0.6e-9, 1.51e-9, 0.3e-9)  050722
#major_xticks = np.arange(0,201,25)  050722

#ax.set_xticks(major_xticks)  050722
#ax.set_yticks(major_yticks)  050722


ax.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)

plt.grid(True)

ax.tick_params(axis='both', labelsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWI_inst_DLS3.svg', dpi=1000,
            bbox_inches = 'tight')

#plt.show()



#%% Instantaneous GWI 

## GRAPH BREAK

from matplotlib.gridspec import GridSpec

##https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib

#mm = (146, 90)  # x value then y value
mm = (160, 100)  # x value then y value
inches = (mm[0] / 25.4, mm[1] / 25.4)

fig = plt.figure(figsize=inches)
#fig.text(0.02, 0.6, r"Y axis label", va="center", rotation="vertical", fontsize=12)
gs = GridSpec(2, 2, height_ratios=[1, 4])

ax_g = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
ax_h = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))


ax_g.plot(t, GWI_inst_tot_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_g.plot(t, GWI_inst_tot_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_g.plot(t, GWI_inst_tot_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_g.plot(t, GWI_inst_tot_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_g.plot(t, GWI_inst_tot_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_g.plot(t, GWI_inst_tot_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_g.plot(t, GWI_inst_tot_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_g.plot(t, GWI_inst_tot_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_g.plot(t, GWI_inst_tot_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_g.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)




ax_h.plot(t, GWI_inst_tot_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_h.plot(t, GWI_inst_tot_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_h.plot(t, GWI_inst_tot_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_h.plot(t, GWI_inst_tot_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_h.plot(t, GWI_inst_tot_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_h.plot(t, GWI_inst_tot_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_h.plot(t, GWI_inst_tot_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_h.plot(t, GWI_inst_tot_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_h.plot(t, GWI_inst_tot_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_h.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)



# zoom-in / limit the view to different portions of the data

ax_g.set_ylim(0.0025, 0.003)
ax_h.set_ylim(-0.001, 0.0008)   

ax_h.set_xlim(0,200)


# hide the spines between ax and ax2
ax_g.spines['bottom'].set_visible(False)
ax_h.spines['top'].set_visible(False)
ax_g.xaxis.tick_top()
ax_g.tick_params(labeltop=False)  # don't put tick labels at the top
ax_h.xaxis.tick_bottom()





ax_g.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)



#https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_g.plot([0, 1], [0, 0], transform=ax_g.transAxes, **kwargs)
ax_h.plot([0, 1], [1, 1], transform=ax_h.transAxes, **kwargs)




ax_g.grid(True)
ax_h.grid(True)

#align grid lines on two plots: https://stackoverflow.com/questions/28692608/align-grid-lines-on-two-plots
from matplotlib.ticker import LinearLocator
N = 5 # Set number of gridlines you want to have in each graph
Y = 2

ax_g.xaxis.set_major_locator(LinearLocator(N))
ax_h.xaxis.set_major_locator(LinearLocator(N))

ax_g.yaxis.set_major_locator(LinearLocator(Y))
#ax_h.yaxis.set_major_locator(LinearLocator(N))




ax_h.set_xlabel('Time (year)', fontsize=14)
ax_h.set_ylabel('GWI_inst (W/(m$^2$.year)', fontsize=14)

ax_g.set_title('Instantaneous GWI, DLS3', fontsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWI_inst_DLS3_break.svg', dpi=1000,
            bbox_inches = 'tight')


plt.draw()




#%% Cumulative GWI

#Step (17): Calculate cumulative global warming impact (GWI)

##Wood-based
GWI_cum_S2_Ac_18y = np.cumsum(GWI_inst_tot_S2_Ac_18y)
GWI_cum_S2_Tgr_60y = np.cumsum(GWI_inst_tot_S2_Tgr_60y)
GWI_cum_E2_Hbr_40y = np.cumsum(GWI_inst_tot_E2_Hbr_40y)


##NonRW
GWI_cum_NonRW_S2_Ac_18y = np.cumsum(GWI_inst_tot_NonRW_S2_Ac_18y)
GWI_cum_NonRW_S2_Tgr_60y = np.cumsum(GWI_inst_tot_NonRW_S2_Tgr_60y)
GWI_cum_NonRW_E2_Hbr_40y = np.cumsum(GWI_inst_tot_NonRW_E2_Hbr_40y)


#Reforestation
GWI_cum_refor_S2_Ac_18y = np.cumsum(GWI_inst_tot_refor_S2_Ac_18y)
GWI_cum_refor_S2_Tgr_60y = np.cumsum(GWI_inst_tot_refor_S2_Tgr_60y)
GWI_cum_refor_E2_Hbr_40y = np.cumsum(GWI_inst_tot_refor_E2_Hbr_40y)

#print(GWI_cum_NonRW_S1_Ac_18y)



fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)


ax.plot(t, GWI_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax.plot(t, GWI_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax.plot(t, GWI_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax.plot(t, GWI_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax.plot(t, GWI_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax.plot(t, GWI_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax.plot(t, GWI_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax.plot(t, GWI_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax.plot(t, GWI_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

ax.set_xlabel('Time (year)', fontsize=14)
#plt.ylabel('GWI_cum (10$^{-10}$ W/m$^2$)')
ax.set_ylabel('GWI_cum (W/(m$^2$))', fontsize=14)


ax.set_xlim(0,200)
#ax.set_ylim(-0.5e-7, 2e-7)  
#ax.set_ylim(-0.1, 0.15)   
ax.set_ylim(-0.1, 0.15)  
 

#major_yticks = np.arange(-0.5e-7, 2.01e-7, 0.5e-7)   050722
#major_xticks = np.arange(0,201,25)  050722

#ax.set_xticks(major_xticks) 050722
#ax.set_yticks(major_yticks) 050722

#plt.fill_between(t, GWI_cum_NonRW_S2_Tgr_60y, GWI_cum_NonRW_S2_Ac_7y, color='lightcoral', alpha=0.3) 

plt.grid(True)

ax.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)

ax.set_title('Cumulative GWI, DLS3', fontsize=14)

ax.tick_params(axis='both', labelsize=14)

plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWIcum_DLS3.svg', dpi=1000,
            bbox_inches = 'tight')

#plt.show()


#%%

#Step (17): Calculate cumulative global warming impact (GWI)

## GRAPH BREAK

from matplotlib.gridspec import GridSpec

##https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib

#mm = (146, 90)  # x value then y value
mm = (160, 100)  # x value then y value
inches = (mm[0] / 25.4, mm[1] / 25.4)

fig = plt.figure(figsize=inches)
#fig.text(0.02, 0.6, r"Y axis label", va="center", rotation="vertical", fontsize=12)
gs = GridSpec(2, 2, height_ratios=[1, 4])

ax_e = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
ax_f = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))



ax_e.plot(t, GWI_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_e.plot(t, GWI_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_e.plot(t, GWI_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_e.plot(t, GWI_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_e.plot(t, GWI_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_e.plot(t, GWI_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_e.plot(t, GWI_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_e.plot(t, GWI_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_e.plot(t, GWI_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)

ax_e.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)





ax_f.plot(t, GWI_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_f.plot(t, GWI_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_f.plot(t, GWI_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_f.plot(t, GWI_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_f.plot(t, GWI_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_f.plot(t, GWI_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_f.plot(t, GWI_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_f.plot(t, GWI_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_f.plot(t, GWI_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)



ax_f.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)



# zoom-in / limit the view to different portions of the data

ax_e.set_ylim(0.25, 0.3)
ax_f.set_ylim(-0.1, 0.08)   

ax_f.set_xlim(0,200)


# hide the spines between ax and ax2
ax_e.spines['bottom'].set_visible(False)
ax_f.spines['top'].set_visible(False)
ax_e.xaxis.tick_top()
ax_e.tick_params(labeltop=False)  # don't put tick labels at the top
ax_f.xaxis.tick_bottom()





ax_e.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)



#https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_e.plot([0, 1], [0, 0], transform=ax_e.transAxes, **kwargs)
ax_f.plot([0, 1], [1, 1], transform=ax_f.transAxes, **kwargs)




ax_e.grid(True)
ax_f.grid(True)

#align grid lines on two plots: https://stackoverflow.com/questions/28692608/align-grid-lines-on-two-plots
from matplotlib.ticker import LinearLocator
N = 5 # Set number of gridlines you want to have in each graph
Y = 2

ax_e.xaxis.set_major_locator(LinearLocator(N))
ax_f.xaxis.set_major_locator(LinearLocator(N))

ax_e.yaxis.set_major_locator(LinearLocator(Y))
#ax_f.yaxis.set_major_locator(LinearLocator(N))

#N = ax_d.xaxis.get_major_ticks()
#ax_c.xaxis.set_major_locator(LinearLocator(N))


ax_f.set_xlabel('Time (year)', fontsize=14)
#ax_c.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)
ax_f.set_ylabel('GWI_cum (W/(m$^2$))', fontsize=14)

ax_e.set_title('Cumulative GWI, DLS3', fontsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWI_cum_DLS3_break.svg', dpi=1000,
            bbox_inches = 'tight')


plt.draw()


#%%

#Step (18): Determine the Instantenous and Cumulative GWI for the  emission reference (1 kg CO2 emission at time zero) before performing dynamic GWP calculation

#determine the GWI inst for the emission reference (1 kg CO2 emission at time zero)

t = np.arange(0,tf-1,1)

matrix_GWI_ref = (tf-1,1)
GWI_inst_ref = np.zeros(matrix_GWI_S2_Ac_18y)

for t in range(0,tf-1):
    GWI_inst_ref[t,0] = np.sum(np.multiply(emission_CO2_ref,DCF_CO2_ti[:,t]))

#print(GWI_inst_ref[:,0])

len(GWI_inst_ref)



#determine the GWI cumulative for the emission reference

t = np.arange(0,tf-1,1)

GWI_cum_ref = np.cumsum(GWI_inst_ref[:,0])
#print(GWI_cum_ref)

#plt.xlabel('Time (year)')
#plt.ylabel('GWI_cum_ref (10$^{-13}$ W/m$^2$.kgCO$_2$)')


#plt.plot(t, GWI_cum_ref)



len(GWI_cum_ref)

#%%

#Step (19): Calculate dynamic global warming potential (GWPdyn)

##Wood-based
GWP_dyn_cum_S2_Ac_18y = [x/(y*1000) for x,y in zip(GWI_cum_S2_Ac_18y, GWI_cum_ref)]
GWP_dyn_cum_S2_Tgr_60y = [x/(y*1000) for x,y in zip(GWI_cum_S2_Tgr_60y, GWI_cum_ref)]
GWP_dyn_cum_E2_Hbr_40y = [x/(y*1000) for x,y in zip(GWI_cum_E2_Hbr_40y, GWI_cum_ref)]

##NonRW
GWP_dyn_cum_NonRW_S2_Ac_18y = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_S2_Ac_18y, GWI_cum_ref)]
GWP_dyn_cum_NonRW_S2_Tgr_60y = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_S2_Tgr_60y, GWI_cum_ref)]
GWP_dyn_cum_NonRW_E2_Hbr_40y = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_E2_Hbr_40y, GWI_cum_ref)]

#Reforestation
GWP_dyn_cum_refor_S2_Ac_18y = [x/(y*1000) for x,y in zip(GWI_cum_refor_S2_Ac_18y, GWI_cum_ref)]
GWP_dyn_cum_refor_S2_Tgr_60y = [x/(y*1000) for x,y in zip(GWI_cum_refor_S2_Tgr_60y, GWI_cum_ref)]
GWP_dyn_cum_refor_E2_Hbr_40y = [x/(y*1000) for x,y in zip(GWI_cum_refor_E2_Hbr_40y, GWI_cum_ref)]


#%%
#normal GWP graph
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)


ax.plot(t, GWP_dyn_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax.plot(t, GWP_dyn_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax.plot(t, GWP_dyn_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax.plot(t, GWP_dyn_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax.plot(t, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax.plot(t, GWP_dyn_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax.plot(t, GWP_dyn_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax.plot(t, GWP_dyn_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax.plot(t, GWP_dyn_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWP_dyn_cum_NonRW_S2_Ac_7y, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='lightcoral', alpha=0.3) 

plt.grid(True)

ax.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)

ax.set_xlim(0,200)
#ax.set_ylim(-500,1500)   050722
#ax.set_ylim(-600,1500)
#ax.set_ylim(-0.7e9,2.1e9)
ax.set_ylim(-0.7e9,1e9)


#major_yticks = np.arange(-500,1501,250)     050722
#major_xticks = np.arange(0,201,25)   050722

#ax.set_xticks(major_xticks)   050722
#ax.set_yticks(major_yticks)   050722

ax.set_xlabel('Time (year)', fontsize=14)
ax.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)

ax.tick_params(axis='both', labelsize=14)

ax.set_title('Dynamic GWP, DLS3', fontsize=14)

plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWPdyn_DLS3.svg', dpi=1000,
            bbox_inches = 'tight')

#%%
#GWP break graph


from matplotlib.gridspec import GridSpec

##https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib

#mm = (146, 90)  # x value then y value
mm = (160, 100)  # x value then y value
inches = (mm[0] / 25.4, mm[1] / 25.4)

fig = plt.figure(figsize=inches)
#fig.text(0.02, 0.6, r"Y axis label", va="center", rotation="vertical", fontsize=12)
gs = GridSpec(2, 2, height_ratios=[1, 4])

ax_c = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
ax_d = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))


ax_c.plot(t, GWP_dyn_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_c.plot(t, GWP_dyn_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_c.plot(t, GWP_dyn_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_c.plot(t, GWP_dyn_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_c.plot(t, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_c.plot(t, GWP_dyn_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_c.plot(t, GWP_dyn_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_c.plot(t, GWP_dyn_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_c.plot(t, GWP_dyn_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_c.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)




ax_d.plot(t, GWP_dyn_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_d.plot(t, GWP_dyn_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_d.plot(t, GWP_dyn_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')


ax_d.plot(t, GWP_dyn_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_d.plot(t, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_d.plot(t, GWP_dyn_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_d.plot(t, GWP_dyn_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_d.plot(t, GWP_dyn_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_d.plot(t, GWP_dyn_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_d.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)



# zoom-in / limit the view to different portions of the data
ax_d.set_xlim(0,200)

ax_d.set_ylim(-0.7e9, 0.7e9)  
ax_c.set_ylim(1.8e9, 2.1e9)

# hide the spines between ax and ax2
ax_c.spines['bottom'].set_visible(False)
ax_d.spines['top'].set_visible(False)
ax_c.xaxis.tick_top()
ax_c.tick_params(labeltop=False)  # don't put tick labels at the top
ax_d.xaxis.tick_bottom()





ax_c.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)



#https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_c.plot([0, 1], [0, 0], transform=ax_c.transAxes, **kwargs)
ax_d.plot([0, 1], [1, 1], transform=ax_d.transAxes, **kwargs)




ax_c.grid(True)
ax_d.grid(True)

#align grid lines on two plots: https://stackoverflow.com/questions/28692608/align-grid-lines-on-two-plots
from matplotlib.ticker import LinearLocator
N = 5 # Set number of gridlines you want to have in each graph
Y = 2
Z = 5


ax_c.xaxis.set_major_locator(LinearLocator(N))
ax_d.xaxis.set_major_locator(LinearLocator(N))

ax_d.yaxis.set_major_locator(LinearLocator(Z))

ax_c.yaxis.set_major_locator(LinearLocator(Y))

#ax_c.yaxis.set_major_locator(LinearLocator(Y))
#ax_d.yaxis.set_major_locator(LinearLocator(N))

#N = ax_d.xaxis.get_major_ticks()
#ax_c.xaxis.set_major_locator(LinearLocator(N))


ax_d.set_xlabel('Time (year)', fontsize=14)
#ax_c.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)
ax_d.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)

ax_c.set_title('Dynamic GWP, DLS3', fontsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\GWPdyn_DLS3_break.svg', dpi=1000,
            bbox_inches = 'tight')


plt.draw()

#%%
#GWP break graph - 2

from matplotlib.gridspec import GridSpec



mm = (146, 90)  # x value then y value
inches = (mm[0] / 25.4, mm[1] / 25.4)

fig = plt.figure(figsize=inches)
#fig.text(0.02, 0.6, r"Y axis label", va="center", rotation="vertical", fontsize=12)
gs = GridSpec(2, 2, height_ratios=[1, 4])

ax_e = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
ax_f = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))
palette = plt.get_cmap("tab20")

ax_e.plot(t, GWP_dyn_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_e.plot(t, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_e.plot(t, GWP_dyn_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_e.plot(t, GWP_dyn_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_e.plot(t, GWP_dyn_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_e.plot(t, GWP_dyn_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_e.plot(t, GWP_dyn_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_e.plot(t, GWP_dyn_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_e.plot(t, GWP_dyn_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_e.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)




ax_f.plot(t, GWP_dyn_cum_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', alpha=0.55)
ax_f.plot(t, GWP_dyn_cum_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', alpha=0.55)
ax_f.plot(t, GWP_dyn_cum_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', alpha=0.55)


ax_f.plot(t, GWP_dyn_cum_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', alpha=0.55)
ax_f.plot(t, GWP_dyn_cum_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', alpha=0.55)
ax_f.plot(t, GWP_dyn_cum_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', alpha=0.55)


ax_f.plot(t, GWP_dyn_cum_S2_Ac_18y, color='forestgreen', label='M_Ac')
ax_f.plot(t, GWP_dyn_cum_S2_Tgr_60y, color='deeppink', label='M_Tgr')
ax_f.plot(t, GWP_dyn_cum_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr')

ax_f.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)



#indx = np.arange(len(data.index))

#labs = data.index.tolist()
#labs.insert(0, "")

ax_e.tick_params(axis="both", which="major", labelsize=10)
ax_f.tick_params(axis="both", which="major", labelsize=10)





# zoom-in / limit the view to different portions of the data
ax_e.set_ylim(1.8e9, 2.1e9)  # outliers only
ax_f.set_ylim(-0.7e9, 0.7e9)  # most of the data

ax_f.set_xlim(0,200)


ax_e.spines["bottom"].set_visible(False)
ax_f.spines["top"].set_visible(False)

ax_e.grid(color="k", alpha=0.5, linestyle=":", zorder=1)
ax_f.grid(color="k", alpha=0.5, linestyle=":", zorder=1)

#ax_c.tick_params(axis="x", which="both", length=0)
#ax_f.tick_params(labeltop="off")
#ax_e.tick_params(labeltop="off")
ax_f.xaxis.tick_bottom()

#d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax_c.transAxes, color="k", clip_on=False)  # linewidth=1)
#ax_c.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
#ax_c.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

#kwargs.update(transform=ax_d.transAxes)  # switch to the bottom axes
#ax_d.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax_d.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal





d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_e.plot([0, 1], [0, 0], transform=ax_e.transAxes, **kwargs)
ax_f.plot([0, 1], [1, 1], transform=ax_f.transAxes, **kwargs)





ax_f.set_xlabel('Time (year)')
#ax_c.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)
ax_f.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)', fontsize=14)

ax_e.set_title('Dynamic GWP, DLS3', fontsize=14)

plt.subplots_adjust(top=0.943, bottom=0.214, left=0.103, right=0.97, hspace=0.133, wspace=0.062)
    
#plt.show()



#%%

#Step (20): Exporting the data behind result graphs to Excel

year = []
for x in range (0, 201): 
    year.append(x) 


### Create Column
    
Col1 = year

##GWI_Inst
#GWI_inst from wood-based scenarios
Col_GI_2 = GWI_inst_tot_S2_Ac_18y
Col_GI_3  = GWI_inst_tot_S2_Tgr_60y
Col_GI_4  = GWI_inst_tot_E2_Hbr_40y



#print(Col_GI_1)
#print(np.shape(Col_GI_1))

#GWI_inst from counter use scenarios
Col_GI_6  = GWI_inst_tot_NonRW_S2_Ac_18y
Col_GI_7  = GWI_inst_tot_NonRW_S2_Tgr_60y
Col_GI_8  = GWI_inst_tot_NonRW_E2_Hbr_40y

#GWI_inst from reforestation scenarios
Col_GI_6_R  = GWI_inst_tot_refor_S2_Ac_18y
Col_GI_7_R  = GWI_inst_tot_refor_S2_Tgr_60y
Col_GI_8_R  = GWI_inst_tot_refor_E2_Hbr_40y

#print(Col_GI_7)
#print(np.shape(Col_GI_7))


#create column results
    
##GWI_cumulative
#GWI_cumulative from wood-based scenarios
Col_GC_2 = GWI_cum_S2_Ac_18y
Col_GC_3 = GWI_cum_S2_Tgr_60y
Col_GC_4 = GWI_cum_E2_Hbr_40y


#GWI_cumulative from counter use scenarios
Col_GC_6 = GWI_cum_NonRW_S2_Ac_18y
Col_GC_7 = GWI_cum_NonRW_S2_Tgr_60y
Col_GC_8 = GWI_cum_NonRW_E2_Hbr_40y

#GWI_cumulative from reforestation scenarios
Col_GC_6_R = GWI_cum_refor_S2_Ac_18y
Col_GC_7_R = GWI_cum_refor_S2_Tgr_60y
Col_GC_8_R = GWI_cum_refor_E2_Hbr_40y



#create column results

##GWPdyn
#GWPdyn from wood-based scenarios
Col_GWP_2 = GWP_dyn_cum_S2_Ac_18y
Col_GWP_3 = GWP_dyn_cum_S2_Tgr_60y
Col_GWP_4 = GWP_dyn_cum_E2_Hbr_40y


#GWPdyn from counter use scenarios
Col_GWP_6 = GWP_dyn_cum_NonRW_S2_Ac_18y
Col_GWP_7 = GWP_dyn_cum_NonRW_S2_Tgr_60y
Col_GWP_8 = GWP_dyn_cum_NonRW_E2_Hbr_40y

#GWPdyn from reforestation scenarios
Col_GWP_6_R = GWP_dyn_cum_refor_S2_Ac_18y
Col_GWP_7_R = GWP_dyn_cum_refor_S2_Tgr_60y
Col_GWP_8_R = GWP_dyn_cum_refor_E2_Hbr_40y



#Create colum results
dfM_GI = pd.DataFrame.from_dict({'Year':Col1, 'M_Ac_18y (W/m2)':Col_GI_2,
                                       'M_Tgr_60y (W/m2)':Col_GI_3, 'E_Hbr_40y (W/m2)':Col_GI_4,
                                       'NR_M_Ac_18y (W/m2)':Col_GI_6,
                                       'NR_M_Tgr_60y (W/m2)':Col_GI_7, 'NR_E_Hbr_40y (W/m2)':Col_GI_8,
                                       'R_M_Ac_18y (W/m2)':Col_GI_6_R,
                                       'R_M_Tgr_60y (W/m2)':Col_GI_7_R, 'RR_E_Hbr_40y (W/m2)':Col_GI_8_R})


dfM_GC = pd.DataFrame.from_dict({'Year':Col1, 'M_Ac_18y (W/m2)':Col_GC_2,
                                       'M_Tgr_60y (W/m2)':Col_GC_3, 'E_Hbr_40y (W/m2)':Col_GC_4,
                                       'NR_M_Ac_18y (W/m2)':Col_GC_6,
                                       'NR_M_Tgr_60y (W/m2)':Col_GC_7, 'NR_E_Hbr_40y (W/m2)':Col_GC_8,
                                       'R_M_Ac_18y (W/m2)':Col_GC_6_R,
                                       'R_M_Tgr_60y (W/m2)':Col_GC_7_R, 'R_E_Hbr_40y (W/m2)':Col_GC_8_R})


dfM_GWPdyn = pd.DataFrame.from_dict({'Year':Col1, 'M_Ac_18y (t-CO2-eq)':Col_GWP_2,
                                        'M_Tgr_60y (t-CO2-eq)':Col_GWP_3, 'E_Hbr_40y (t-CO2-eq)':Col_GWP_4,
                                        'NR_M_Ac_18y (t-CO2-eq)':Col_GWP_6,
                                        'NR_M_Tgr_60y (t-CO2-eq)':Col_GWP_7, 'NR_E_Hbr_40y (t-CO2-eq)':Col_GWP_8,
                                        'R_M_Ac_18y (t-CO2-eq)':Col_GWP_6_R,
                                        'R_M_Tgr_60y (t-CO2-eq)':Col_GWP_7_R, 'R_E_Hbr_40y (t-CO2-eq)':Col_GWP_8_R})


#Export to excel
writer = pd.ExcelWriter('GraphResults_DLS3.xlsx', engine = 'xlsxwriter')


dfM_GI.to_excel(writer, sheet_name = 'GWI_Inst_DLS3', header=True, index=False )


dfM_GC.to_excel(writer, sheet_name = 'Cumulative GWI_DLS3', header=True, index=False )


dfM_GWPdyn.to_excel(writer, sheet_name = 'GWPdyn_DL_DLS3', header=True, index=False )


writer.save()


#%%

#Step (21): Generate the excel file for the individual carbon emission and sequestration flows


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)





division = 1000*44/12
division_CH4 = 1000*16/12



##M_Ac_18y
c_firewood_energy_S2_Ac18 = [x/division for x in c_firewood_energy_S2_Ac18]
decomp_tot_S2_Ac_18y[:,0] = [x/division for x in decomp_tot_S2_Ac_18y[:,0]]
TestDSM2_Ac18.o = [x/division for x in TestDSM2_Ac18.o]
PH_Emissions_HWP2_Ac_18y = [x/division for x in PH_Emissions_HWP2_Ac_18y]
#OC_storage_S2_Ac18 = [x/division for x in OC_storage_S2_Ac18]
flat_list_Ac_18y = [x/division for x in flat_list_Ac_18y]
decomp_tot_CO2_S2_Ac_18y[:,0] = [x/division for x in decomp_tot_CO2_S2_Ac_18y[:,0]]

decomp_tot_CH4_S2_Ac_18y[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2_Ac_18y[:,0]]


#M_Ac_18y, NonRW/Counter use
##NonRW emissions (F9-0-2)
emissions_NonRW_S2_Ac_18y = [x/division for x in emissions_NonRW_S2_Ac_18y]

#M_Ac_18y, Reforestation
emissions_refor_S2_Ac_18y = [x/division for x in emissions_refor_S2_Ac_18y]
emissions_refor_S2_Ac_18y_seq = [x/division for x in emissions_refor_S2_Ac_18y_seq]




##M_Tgr_60y
c_firewood_energy_S2_Tgr60 = [x/division for x in c_firewood_energy_S2_Tgr60]
decomp_tot_S2_Tgr_60y[:,0] = [x/division for x in decomp_tot_S2_Tgr_60y[:,0]]
TestDSM2_Tgr60.o = [x/division for x in TestDSM2_Tgr60.o]
PH_Emissions_HWP2_Tgr_60y = [x/division for x in PH_Emissions_HWP2_Tgr_60y]
#OC_storage_S2_Tgr60 = [x/division for x in OC_storage_S2_Tgr60]
flat_list_Tgr_60y = [x/division for x in flat_list_Tgr_60y]
decomp_tot_CO2_S2_Tgr_60y[:,0] = [x/division for x in decomp_tot_CO2_S2_Tgr_60y[:,0]]

decomp_tot_CH4_S2_Tgr_60y[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2_Tgr_60y[:,0]]


#M_Tgr_60y, NonRW/Counter use
##NonRW emissions (F9-0-2)
emissions_NonRW_S2_Tgr_60y = [x/division for x in emissions_NonRW_S2_Tgr_60y]

#M_Ac_Tgr_60y, Reforestation
emissions_refor_S2_Tgr_60y = [x/division for x in emissions_refor_S2_Tgr_60y]
emissions_refor_S2_Tgr_60y_seq = [x/division for x in emissions_refor_S2_Tgr_60y_seq]





##E_Hbr_40y
c_firewood_energy_E2_Hbr40 = [x/division for x in c_firewood_energy_E2_Hbr40]
c_pellets_Hbr_40y = [x/division for x in c_pellets_Hbr_40y]
decomp_tot_E2_Hbr_40y[:,0] = [x/division for x in decomp_tot_E2_Hbr_40y[:,0]]
TestDSME2_Hbr40.o = [x/division for x in TestDSME2_Hbr40.o]
PH_Emissions_HWPE2_Hbr_40y = [x/division for x in PH_Emissions_HWPE2_Hbr_40y]
#OC_storage_E2_Hbr40 = [x/division for x in OC_storage_E2_Hbr40]
flat_list_Hbr_40y = [x/division for x in flat_list_Hbr_40y]
decomp_tot_CO2_E2_Hbr_40y[:,0] = [x/division for x in decomp_tot_CO2_E2_Hbr_40y[:,0]]

decomp_tot_CH4_E2_Hbr_40y[:,0] = [x/division_CH4 for x in decomp_tot_CH4_E2_Hbr_40y[:,0]]


#E_Hbr_40y, NonRW/Counter use
##NonRW emissions (F9-0-2)
emissions_NonRW_E2_Hbr_40y = [x/division for x in emissions_NonRW_E2_Hbr_40y]


#E_Hbr_40y, Reforestation
emissions_refor_E2_Hbr_40y = [x/division for x in emissions_refor_E2_Hbr_40y]
emissions_refor_E2_Hbr_40y_seq = [x/division for x in emissions_refor_E2_Hbr_40y_seq]


#landfill aggregate flows
Landfill_decomp_DL_FP_S2_Ac_18y = decomp_tot_CH4_S2_Ac_18y, decomp_tot_CO2_S2_Ac_18y
Landfill_decomp_DL_FP_S2_Tgr_60y = decomp_tot_CH4_S2_Tgr_60y, decomp_tot_CO2_S2_Tgr_60y
Landfill_decomp_DL_FP_E2_Hbr_40y = decomp_tot_CH4_E2_Hbr_40y, decomp_tot_CO2_E2_Hbr_40y


Landfill_decomp_DL_FP_S2_Ac_18y = [sum(x) for x in zip(*Landfill_decomp_DL_FP_S2_Ac_18y)]
Landfill_decomp_DL_FP_S2_Tgr_60y = [sum(x) for x in zip(*Landfill_decomp_DL_FP_S2_Tgr_60y)]
Landfill_decomp_DL_FP_E2_Hbr_40y = [sum(x) for x in zip(*Landfill_decomp_DL_FP_E2_Hbr_40y)]

Landfill_decomp_DL_FP_S2_Ac_18y = [item for sublist in Landfill_decomp_DL_FP_S2_Ac_18y for item in sublist]
Landfill_decomp_DL_FP_S2_Tgr_60y = [item for sublist in Landfill_decomp_DL_FP_S2_Tgr_60y for item in sublist]
Landfill_decomp_DL_FP_E2_Hbr_40y = [item for sublist in Landfill_decomp_DL_FP_E2_Hbr_40y for item in sublist]


Column1 = year


#M_Ac_18y
Column8 = c_firewood_energy_S2_Ac18
Column9 = decomp_tot_S2_Ac_18y[:,0]
Column10 = TestDSM2_Ac18.o
Column11 = PH_Emissions_HWP2_Ac_18y
#Column12_1 = OC_storage_S2_Ac18
Column12 = Landfill_decomp_DL_FP_S2_Ac_18y
Column13 = flat_list_Ac_18y
Column13_R = emissions_refor_S2_Ac_18y_seq
Column13_NR = emissions_NonRW_S2_Ac_18y


#M_Tgr_60y
Column14 = c_firewood_energy_S2_Tgr60
Column15 = decomp_tot_S2_Tgr_60y[:,0]
Column16 = TestDSM2_Tgr60.o
Column17 = PH_Emissions_HWP2_Tgr_60y 
#Column18_1 = OC_storage_S2_Tgr60
Column18 = Landfill_decomp_DL_FP_S2_Tgr_60y
Column19 = flat_list_Tgr_60y
Column19_R = emissions_refor_S2_Tgr_60y_seq 
Column19_NR = emissions_NonRW_S2_Tgr_60y


#E_Hbr_40y
Column20 = c_firewood_energy_E2_Hbr40
Column20_1 = c_pellets_Hbr_40y
Column21 = decomp_tot_E2_Hbr_40y[:,0]
Column22 = TestDSME2_Hbr40.o
Column23 = PH_Emissions_HWPE2_Hbr_40y
#Column24_1 = OC_storage_E2_Hbr40
Column24 = Landfill_decomp_DL_FP_E2_Hbr_40y
Column25 = flat_list_Hbr_40y
Column25_R = emissions_refor_E2_Hbr_40y_seq
Column25_NR = emissions_NonRW_E2_Hbr_40y



#create columns
dfM_Ac_18y = pd.DataFrame.from_dict({'Year':Column1,'F0-1c: Biomass plantation C sequestration (t-C)':Column13,
                                     #'9: Landfill storage (t-C)':Column12_1,
                                     'F0-1a: Reforestation C sequestration (t-C)': Column13_R,
                                     'F1-0: Residue decomposition (t-C)':Column9,
                                     'F6-0-1: Emissions from firewood/other energy use (t-C)':Column8,
                                     'F8-0: Operational stage/processing emissions (t-C)':Column11,
                                     'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column10,
                                     'F7-0: Landfill gas decomposition (t-C)':Column12,
                                     'F9-0: Non-renewable materials/energy emissions (t_C)': Column13_NR})

dfE_Tgr_60y = pd.DataFrame.from_dict({'Year':Column1,'F0-1c: Biomass plantation C sequestration (t-C)':Column19,
                                      #'9: Landfill storage (t-C)':Column18_1,
                                      'F0-1a: Reforestation C sequestration (t-C)': Column19_R,
                                      'F1-0: Residue decomposition (t-C)':Column15,
                                      'F6-0-1: Emissions from firewood/other energy use (t-C)':Column14,
                                      'F8-0: Operational stage/processing emissions (t-C)':Column17,
                                      'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column16,
                                      'F7-0: Landfill gas decomposition (t-C)':Column18,
                                      'F9-0: Non-renewable materials/energy emissions (t_C)': Column19_NR})

dfE_Hbr_40y = pd.DataFrame.from_dict({'Year':Column1,'F0-1c: Biomass plantation C sequestration (t-C)':Column25,
                                      #'9: Landfill storage (t-C)':Column24_1,
                                      'F0-1a: Reforestation C sequestration (t-C)': Column25_R,
                                      'F1-0: Residue decomposition (t-C)':Column21,
                                      'F6-0-1: Emissions from firewood/other energy use (t-C)':Column20, 
                                      'F8-0: Operational stage/processing emissions (t-C)':Column23,
                                      'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column22,
                                      'F7-0: Landfill gas decomposition (t-C)':Column24,
                                      'F4-0: Emissions from wood pellets use (t-C)':Column20_1,
                                      'F9-0: Non-renewable materials/energy emissions (t_C)': Column25_NR})
    

writer = pd.ExcelWriter('C_flows_DLS3.xlsx', engine = 'xlsxwriter')


dfM_Ac_18y.to_excel(writer, sheet_name = 'DLS3_45%_Ac_18y', header=True, index=False)
dfE_Tgr_60y.to_excel(writer, sheet_name = 'DLS3_45%_Tgr', header=True, index=False)
dfE_Hbr_40y.to_excel(writer, sheet_name = 'DLS3_60%_Hbr', header=True, index=False)



writer.save()



#%%

#Step (22): Plot of the individual carbon emission and sequestration flows for normal and symlog-scale graphs


#plot for the individual carbon flows - test for symlog-scale graphs

#DLS3_45%_Ac_18y

fig=plt.figure()
fig.show()
ax2_s=fig.add_subplot(111)

#plot
ax2_s.plot(t, flat_list_Ac_18y, color='darkkhaki', label='F0-1c: Biomass plantation C sequestration') 
#ax2_s.plot(t, OC_storage_S2_Ac18, color='darkturquoise', label='9: Landfill storage')
ax2_s.plot(t, decomp_tot_S2_Ac_18y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2_s.plot(t, c_firewood_energy_S2_Ac18, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax2_s.plot(t, PH_Emissions_HWP2_Ac_18y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax2_s.plot(t, TestDSM2_Ac18.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax2_s.plot(t, Landfill_decomp_DL_FP_S2_Ac_18y, color='yellow', label='F7-0: Landfill gas decomposition') 


#NonRW plot
ax2_s.plot(t, emissions_NonRW_S2_Ac_18y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax2_s.plot(t, emissions_refor_S2_Ac_18y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)




ax2_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2_s.set_xlim(-1,200)


ax2_s.set_yscale('symlog')
 
ax2_s.set_xlabel('Time (year)')
ax2_s.set_ylabel('C flows (t-C) (symlog)')

ax2_s.set_title('Carbon flow, DLS3_45%_Ac_18y (symlog-scale)')

#plt.show()

#%%

#plot for the individual carbon flows

#DLS3_45%_Ac_18y

fig=plt.figure()
fig.show()
ax2=fig.add_subplot(111)

#plot
ax2.plot(t, flat_list_Ac_18y, color='darkkhaki', label='F0-1c: Biomass C sequestration') 
#ax2.plot(t, OC_storage_S2_Ac18, color='darkturquoise', label='9: Landfill storage')
ax2.plot(t, decomp_tot_S2_Ac_18y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2.plot(t, c_firewood_energy_S2_Ac18, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax2.plot(t, PH_Emissions_HWP2_Ac_18y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax2.plot(t, TestDSM2_Ac18.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax2.plot(t, Landfill_decomp_DL_FP_S2_Ac_18y, color='yellow', label='F7-0: Landfill gas decomposition') 

#NonRW plot
ax2.plot(t, emissions_NonRW_S2_Ac_18y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax2.plot(t, emissions_refor_S2_Ac_18y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)


 
ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2.set_xlim(0,200)


ax2.set_xlabel('Time (year)')
ax2.set_ylabel('C flows(t-C)')

ax2.set_title('Carbon flow, DLS3_45%_Ac_18y')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_M')


plt.draw()

#%%
#plot for the individual carbon flows - test for symlog-scale graphs

#DLS3_45%_Tgr

fig=plt.figure()
fig.show()
ax3_s=fig.add_subplot(111)

#plot
ax3_s.plot(t, flat_list_Tgr_60y, color='darkkhaki', label='F0-1c: Biomass C sequestration') 
#ax3_s.plot(t, OC_storage_S2_Tgr60, color='darkturquoise', label='9: Landfill storage')
ax3_s.plot(t, decomp_tot_S2_Tgr_60y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax3_s.plot(t, c_firewood_energy_S2_Tgr60, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax3_s.plot(t, PH_Emissions_HWP2_Tgr_60y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax3_s.plot(t, TestDSM2_Tgr60.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax3_s.plot(t, Landfill_decomp_DL_FP_S2_Tgr_60y, color='yellow', label='F7-0: Landfill gas decomposition') 


#NonRW plot
ax3_s.plot(t, emissions_NonRW_S2_Tgr_60y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax3_s.plot(t, emissions_refor_S2_Tgr_60y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)



ax3_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax3_s.set_xlim(-1,200)


ax3_s.set_yscale('symlog')
 
ax3_s.set_xlabel('Time (year)')
ax3_s.set_ylabel('C flows (t-C) (symlog)')

ax3_s.set_title('Carbon flow, DLS3_45%_Tgr (symlog-scale)')

#plt.show()


#%%

#plot for the individual carbon flows

#DLS3_45%_Tgr

fig=plt.figure()
fig.show()
ax3=fig.add_subplot(111)

#plot
ax3.plot(t, flat_list_Tgr_60y, color='darkkhaki', label='F0-1c: Biomass C sequestration') 
#ax3.plot(t, OC_storage_S2_Tgr60, color='darkturquoise', label='9: Landfill storage')
ax3.plot(t, decomp_tot_S2_Tgr_60y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax3.plot(t, c_firewood_energy_S2_Tgr60, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax3.plot(t, PH_Emissions_HWP2_Tgr_60y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax3.plot(t, TestDSM2_Tgr60.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax3.plot(t, Landfill_decomp_DL_FP_S2_Tgr_60y, color='yellow', label='F7-0: Landfill gas decomposition') 


#NonRW plot
ax3.plot(t, emissions_NonRW_S2_Tgr_60y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax3.plot(t, emissions_refor_S2_Tgr_60y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)


 
ax3.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax3.set_xlim(0,200)


ax3.set_xlabel('Time (year)')
ax3.set_ylabel('C flows(t-C)')

ax3.set_title('Carbon flow, DLS3_45%_Tgr')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_M')


plt.draw()

#%%
#plot for the individual carbon flows - test for symlog-scale graphs

#DLS3_60%_Hbr

fig=plt.figure()
fig.show()
ax4_s=fig.add_subplot(111)

#plot
ax4_s.plot(t, flat_list_Hbr_40y, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax4_s.plot(t, OC_storage_E2_Hbr40, color='darkturquoise', label='9: Landfill storage')
ax4_s.plot(t, decomp_tot_E2_Hbr_40y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax4_s.plot(t, c_firewood_energy_E2_Hbr40, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax4_s.plot(t, PH_Emissions_HWPE2_Hbr_40y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax4_s.plot(t, Landfill_decomp_DL_FP_E2_Hbr_40y, color='yellow', label='F7-0: Landfill gas decomposition') 
ax4_s.plot(t, c_pellets_Hbr_40y, color='slategrey', label='F4-0: Emissions from wood pellets use') 


#NonRW plot
ax4_s.plot(t, emissions_NonRW_E2_Hbr_40y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax4_s.plot(t, emissions_refor_E2_Hbr_40y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)




ax4_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax4_s.set_xlim(-1,200)


ax4_s.set_yscale('symlog')
 
ax4_s.set_xlabel('Time (year)')
ax4_s.set_ylabel('C flows (t-C) (symlog)')

ax4_s.set_title('Carbon flow, DLS3_60%_Hbr (symlog-scale)')

#plt.show()


#%%

#plot for the individual carbon flows

#DLS3_60%_Hbr

fig=plt.figure()
fig.show()
ax4=fig.add_subplot(111)

#plot
ax4.plot(t, flat_list_Hbr_40y, color='darkkhaki', label='F0-1c: Biomass C sequestration') 
#ax4.plot(t, OC_storage_E2_Hbr40, color='darkturquoise', label='9: Landfill storage')
ax4.plot(t, decomp_tot_E2_Hbr_40y[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax4.plot(t, c_firewood_energy_E2_Hbr40, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax4.plot(t, PH_Emissions_HWPE2_Hbr_40y, color='orange', label='F8-0: Operational stage/processing emissions') 
ax4.plot(t, Landfill_decomp_DL_FP_E2_Hbr_40y, color='yellow', label='F7-0: Landfill gas decomposition') 
ax4.plot(t, c_pellets_Hbr_40y, color='slategrey', label='F4-0: Emissions from wood pellets use') 


#NonRW plot
ax4.plot(t, emissions_NonRW_E2_Hbr_40y, color='cornflowerblue', label='F9-0: Non-renewable materials/energy emissions (t_C)', ls='-.', alpha=0.55)


#Reforestation plot
ax4.plot(t, emissions_refor_E2_Hbr_40y_seq, color='forestgreen', label='F0-1a: Reforestation C sequestration (t-C)', ls=':', alpha=0.55)
 


ax4.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax4.set_xlim(0,200)


ax4.set_xlabel('Time (year)')
ax4.set_ylabel('C flows(t-C)')

ax4.set_title('Carbon flow, DLS3_60%_Hbr')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_M')


plt.draw()

#%%

#Step (23): Generate the excel file for the net carbon balance


Agg_Cflow_S2_Ac_18y = [c_firewood_energy_S2_Ac18, decomp_tot_S2_Ac_18y[:,0], TestDSM2_Ac18.o, PH_Emissions_HWP2_Ac_18y, Landfill_decomp_DL_FP_S2_Ac_18y, flat_list_Ac_18y]
Agg_Cflow_S2_Tgr_60y = [c_firewood_energy_S2_Tgr60, decomp_tot_S2_Tgr_60y[:,0], TestDSM2_Tgr60.o, PH_Emissions_HWP2_Tgr_60y, Landfill_decomp_DL_FP_S2_Tgr_60y, flat_list_Tgr_60y]
Agg_Cflow_E2_Hbr_40y = [c_firewood_energy_E2_Hbr40, c_pellets_Hbr_40y, decomp_tot_E2_Hbr_40y[:,0], TestDSME2_Hbr40.o, PH_Emissions_HWPE2_Hbr_40y, Landfill_decomp_DL_FP_E2_Hbr_40y, flat_list_Hbr_40y]


Agg_Cflow_DL_FP_S2_Ac_18y = [sum(x) for x in zip(*Agg_Cflow_S2_Ac_18y)]
Agg_Cflow_DL_FP_S2_Tgr_60y = [sum(x) for x in zip(*Agg_Cflow_S2_Tgr_60y)]
Agg_Cflow_DL_FP_E2_Hbr_40y = [sum(x) for x in zip(*Agg_Cflow_E2_Hbr_40y)]


##Counter use/NR scenarios
Agg_Cflow_NonRW_S2_Ac_18y = emissions_NonRW_S2_Ac_18y
Agg_Cflow_NonRW_S2_Tgr_60y = emissions_NonRW_S2_Tgr_60y
Agg_Cflow_NonRW_E2_Hbr_40y = emissions_NonRW_E2_Hbr_40y


##Reforestation scenarions
Agg_Cflow_refor_S2_Ac_18y = [emissions_refor_S2_Ac_18y, emissions_refor_S2_Ac_18y_seq]
Agg_Cflow_refor_S2_Tgr_60y = [emissions_refor_S2_Tgr_60y, emissions_refor_S2_Tgr_60y_seq]
Agg_Cflow_refor_E2_Hbr_40y = [emissions_refor_E2_Hbr_40y, emissions_refor_E2_Hbr_40y_seq]


Agg_Cflow_refor_S2_Ac_18y = [sum(x) for x in zip(*Agg_Cflow_refor_S2_Ac_18y)]
Agg_Cflow_refor_S2_Tgr_60y = [sum(x) for x in zip(*Agg_Cflow_refor_S2_Tgr_60y)]
Agg_Cflow_refor_E2_Hbr_40y = [sum(x) for x in zip(*Agg_Cflow_refor_E2_Hbr_40y)]




#create column year
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)

#Create colum results
dfM_DLS3 = pd.DataFrame.from_dict({'Year':year, 'M_Ac_18y (t-C)':Agg_Cflow_DL_FP_S2_Ac_18y,
                                         'M_Tgr_60y (t-C)':Agg_Cflow_DL_FP_S2_Tgr_60y, 'E_Hbr_40y (t-C)':Agg_Cflow_DL_FP_E2_Hbr_40y,
                                         'NR_M_Ac_18y (t-C)':Agg_Cflow_NonRW_S2_Ac_18y,
                                         'NR_M_Tgr_60y (t-C)':Agg_Cflow_NonRW_S2_Tgr_60y, 'NR_E_Hbr_40y (t-C)':Agg_Cflow_NonRW_E2_Hbr_40y,
                                         'R_M_Ac_18y (t-C)':Agg_Cflow_refor_S2_Ac_18y,
                                         'R_M_Tgr_60y (t-C)':Agg_Cflow_refor_S2_Tgr_60y, 'R_E_Hbr_40y (t-C)':Agg_Cflow_refor_E2_Hbr_40y})

    
#Export to excel
writer = pd.ExcelWriter('AggCFlow_DLS3.xlsx', engine = 'xlsxwriter')


dfM_DLS3.to_excel(writer, sheet_name = 'DLS3', header=True, index=False)

writer.save()



#%%

#Step (24): Plot the net carbon balance 


fig=plt.figure()
fig.show()
ax5=fig.add_subplot(111)



ax5.plot(t, Agg_Cflow_DL_FP_S2_Ac_18y, color='forestgreen', label='M_Ac', linewidth=0.9) 
ax5.plot(t, Agg_Cflow_DL_FP_S2_Tgr_60y, color='deeppink', label='M_Tgr', linewidth=0.9)
ax5.plot(t, Agg_Cflow_DL_FP_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr', linewidth=0.9)  


#NonRW plot
ax5.plot(t, Agg_Cflow_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', linewidth=1.2)
ax5.plot(t, Agg_Cflow_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', linewidth=1.2)
ax5.plot(t, Agg_Cflow_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', linewidth=1.2)


#Reforestation plot
ax5.plot(t, Agg_Cflow_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', linewidth=3.5)
ax5.plot(t, Agg_Cflow_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':', linewidth=3.5)
ax5.plot(t, Agg_Cflow_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':', linewidth=3.5)



#ax5.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

ax5.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)

ax5.tick_params(axis='both', labelsize=14)

#ax5.set_xlim(-0.35,85)
ax5.set_xlim(-0.35,85)
#ax5.set_ylim(-0.5e8,3.5e8)
ax5.set_ylim(-0.5e8,1e8)

plt.yticks(np.arange(-0.5e8, 1.1e8, 0.5e8))

#ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#ax5.set_yscale('symlog')
 
ax5.set_xlabel('Time (year)', fontsize=14)
ax5.set_ylabel('C flows (t-C)', fontsize=14)

ax5.set_title('Net Carbon Balance, DLS3', fontsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\_NCB_DLS3.svg', dpi=1000,
            bbox_inches = 'tight')


#plt.show()

#%%

#Step (24): Plot the net carbon balance 


## GRAPH BREAK

from matplotlib.gridspec import GridSpec

##https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib

#mm = (146, 90)  # x value then y value
mm = (160, 100)  # x value then y value
inches = (mm[0] / 25.4, mm[1] / 25.4)

fig = plt.figure(figsize=inches)
#fig.text(0.02, 0.6, r"Y axis label", va="center", rotation="vertical", fontsize=12)
gs = GridSpec(2, 2, height_ratios=[1, 4])

ax_i = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
ax_j = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))


ax_i.plot(t, Agg_Cflow_DL_FP_S2_Ac_18y, color='forestgreen', label='M_Ac', linewidth=0.9) 
ax_i.plot(t, Agg_Cflow_DL_FP_S2_Tgr_60y, color='deeppink', label='M_Tgr', linewidth=0.9)
ax_i.plot(t, Agg_Cflow_DL_FP_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr', linewidth=0.9)  

#NonRW plot
ax_i.plot(t, Agg_Cflow_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls=':', linewidth=3.5)
ax_i.plot(t, Agg_Cflow_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls=':', linewidth=3.5)
ax_i.plot(t, Agg_Cflow_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls=':', linewidth=3.5)

#Reforestation plot
ax_i.plot(t, Agg_Cflow_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls='-.', linewidth=1.2)
ax_i.plot(t, Agg_Cflow_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls='-.',linewidth=1.2)
ax_i.plot(t, Agg_Cflow_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls='-.',linewidth=1.2)


#ax_i.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)





ax_j.plot(t, Agg_Cflow_DL_FP_S2_Ac_18y, color='forestgreen', label='M_Ac', linewidth=0.9) 
ax_j.plot(t, Agg_Cflow_DL_FP_S2_Tgr_60y, color='deeppink', label='M_Tgr', linewidth=0.9)
ax_j.plot(t, Agg_Cflow_DL_FP_E2_Hbr_40y, color='cornflowerblue', label='E_Hbr', linewidth=0.9)  

#NonRW plot
ax_j.plot(t, Agg_Cflow_NonRW_S2_Ac_18y, color='forestgreen', label='CU_M_Ac', ls='-.', linewidth=1.2)
ax_j.plot(t, Agg_Cflow_NonRW_S2_Tgr_60y, color='deeppink', label='CU_M_Tgr', ls='-.', linewidth=1.2)
ax_j.plot(t, Agg_Cflow_NonRW_E2_Hbr_40y, color='cornflowerblue', label='CU_E_Hbr', ls='-.', linewidth=1.2)

#Reforestation plot
ax_j.plot(t, Agg_Cflow_refor_S2_Ac_18y, color='forestgreen', label='R_M_Ac', ls=':', linewidth=3.5)
ax_j.plot(t, Agg_Cflow_refor_S2_Tgr_60y, color='deeppink', label='R_M_Tgr', ls=':',linewidth=3.5)
ax_j.plot(t, Agg_Cflow_refor_E2_Hbr_40y, color='cornflowerblue', label='R_E_Hbr', ls=':',linewidth=3.5)

#ax_j.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)


# zoom-in / limit the view to different portions of the data

ax_i.set_ylim(3e8, 3.5e8)
ax_j.set_ylim(-0.5e8, 0.6e8)   

ax_j.set_xlim(-0.35,85)


# hide the spines between ax and ax2
ax_i.spines['bottom'].set_visible(False)
ax_j.spines['top'].set_visible(False)
ax_i.xaxis.tick_top()
ax_i.tick_params(labeltop=False)  # don't put tick labels at the top
ax_j.xaxis.tick_bottom()





ax_i.legend(bbox_to_anchor=(0.99,1), loc="upper left", prop={'size': 14}, frameon=False)



#https://stackoverflow.com/questions/59305080/formatting-a-broken-y-axis-in-python-matplotlib
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_i.plot([0, 1], [0, 0], transform=ax_i.transAxes, **kwargs)
ax_j.plot([0, 1], [1, 1], transform=ax_j.transAxes, **kwargs)




#ax_i.grid(True)
#ax_j.grid(True)

#align grid lines on two plots: https://stackoverflow.com/questions/28692608/align-grid-lines-on-two-plots
#from matplotlib.ticker import LinearLocator
#N = 5 # Set number of gridlines you want to have in each graph
#Y = 2

#ax_i.xaxis.set_major_locator(LinearLocator(N))
#ax_j.xaxis.set_major_locator(LinearLocator(N))

#ax_i.yaxis.set_major_locator(LinearLocator(Y))
#ax_j.yaxis.set_major_locator(LinearLocator(N))




ax_j.set_xlabel('Time (year)', fontsize=14)
ax_j.set_ylabel('C flows (t-C)', fontsize=14)

ax_i.set_title('Net Carbon Balance, DLS3', fontsize=14)


plt.savefig('C:\Work\Data\ID Future Scenarios\_National-scale\Fig_Manuscript\_NCB_DLS3_break.svg', dpi=1000,
            bbox_inches = 'tight')


plt.draw()


#%%

#Step (25): Generate the excel file for documentation of individual carbon flows in the system definition (Fig. 1)


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)


df2_Ac18 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Ac_18y')
df2_Tgr60 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_45%_Tgr')
dfE2_Hbr40 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\DLS3.xlsx', 'DLS3_60%_Hbr')



Column1 = year
division = 1000*44/12
division_CH4 = 1000*16/12


    
    
##S2_Ac_18y
## define the input flow for the landfill (F5-7)
OC_storage_S2_Ac18 = df2_Ac18['Other_C_storage'].values


OC_storage_S2_Ac18 = [x/division for x in OC_storage_S2_Ac18]
OC_storage_S2_Ac18 = [abs(number) for number in OC_storage_S2_Ac18]

C_LF_S2_Ac18 = [x*1/0.82 for x in OC_storage_S2_Ac18]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_S2_Ac18 = [x/division for x in df2_Ac18['Input_PF'].values]
HWP_S2_Ac18_energy =  [x*1/3 for x in c_firewood_energy_S2_Ac18]
HWP_S2_Ac18_landfill = [x*1/0.82 for x in OC_storage_S2_Ac18]

HWP_S2_Ac18_sum = [HWP_S2_Ac18, HWP_S2_Ac18_energy, HWP_S2_Ac18_landfill]
HWP_S2_Ac18_sum = [sum(x) for x in zip(*HWP_S2_Ac18_sum )]

## in-use stocks (S-4)
TestDSM2_Ac18.s = [x/division for x in TestDSM2_Ac18.s]
#TestDSM2_Ac18.i = [x/division for x in TestDSM2_Ac18.i]



#calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_S2_Ac_18y = (tf,1)
stocks_S2_Ac_18y = np.zeros(zero_matrix_stocks_S2_Ac_18y)


i = 0
stocks_S2_Ac_18y[0] = C_LF_S2_Ac18[0] - Landfill_decomp_DL_FP_S2_Ac_18y[0]

while i < tf-1:
    stocks_S2_Ac_18y[i+1] = np.array(C_LF_S2_Ac18[i+1] - Landfill_decomp_DL_FP_S2_Ac_18y[i+1] + stocks_S2_Ac_18y[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_S2_Ac_18y = [x1+x2 for (x1,x2) in zip(HWP_S2_Ac18_sum, [x*2/3 for x in c_firewood_energy_S2_Ac18])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2_Ac_18y = (tf,1)
ForCstocks_S2_Ac_18y = np.zeros(zero_matrix_ForCstocks_S2_Ac_18y)

i = 0
ForCstocks_S2_Ac_18y[0] = initAGB - flat_list_Ac_18y[0] - decomp_tot_S2_Ac_18y[0] - HWP_logged_S2_Ac_18y[0]

while i < tf-1:
    ForCstocks_S2_Ac_18y[i+1] = np.array(ForCstocks_S2_Ac_18y[i] - flat_list_Ac_18y[i+1] - decomp_tot_S2_Ac_18y[i+1] - HWP_logged_S2_Ac_18y[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2_amount_Ac18 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\NonRW_DLS3.xlsx', 'DLS3_45%_Ac_18y')
NonRW_amount_S2_Ac_18y = df2_amount_Ac18['NonRW_amount'].values

NonRW_amount_S2_Ac_18y = [x/1000 for x in NonRW_amount_S2_Ac_18y]





#create columns
dfM_Ac_18y = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1c (t-C)': flat_list_Ac_18y,
                                    'F0-1a (t-C)': emissions_refor_S2_Ac_18y_seq,
                                    'F1-0 (t-C)': decomp_tot_S2_Ac_18y[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Ac_18y,
                                    #'F1c-2 (t-C)': FP_S2_Ac_18y,
                                    'F1-2 (t-C)': HWP_logged_S2_Ac_18y, 
                                    'St-1 (t-C)':ForCstocks_S2_Ac_18y[:,0], 
                                    'F2-3 (t-C)': HWP_S2_Ac18_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2_Ac18], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2_Ac18_sum, [x*1/0.82 for x in OC_storage_S2_Ac18], [x*1/3 for x in c_firewood_energy_S2_Ac18])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2_Ac18],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2_Ac18], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2_Ac18.s, 
                                    #'S-4-i (t-C)': TestDSM2_Ac7.i,
                                    'F4-5 (t-C)': TestDSM2_Ac18.o,
                                    'F5-6 (t-C)': TestDSM2_Ac18.o, 
                                    'F5-7 (t-C)': C_LF_S2_Ac18,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2_Ac18,
                                    'F6-0-2 (t-C)': TestDSM2_Ac18.o,
                                    'St-7 (t-C)': stocks_S2_Ac_18y[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_DL_FP_S2_Ac_18y,
                                    'F8-0 (t-C)': PH_Emissions_HWP2_Ac_18y,
                                    'S9-0 (t)': NonRW_amount_S2_Ac_18y, 
                                    'F9-0 (t-C)': emissions_NonRW_S2_Ac_18y,
                                    })

    
##S2_Tgr_60y
## define the input flow for the landfill (F5-7)
OC_storage_S2_Tgr60 = df2_Tgr60['Other_C_storage'].values


OC_storage_S2_Tgr60 = [x/division for x in OC_storage_S2_Tgr60]
OC_storage_S2_Tgr60 = [abs(number) for number in OC_storage_S2_Tgr60]

C_LF_S2_Tgr60 = [x*1/0.82 for x in OC_storage_S2_Tgr60]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_S2_Tgr60 = [x/division for x in df2_Tgr60['Input_PF'].values]
HWP_S2_Tgr60_energy =  [x*1/3 for x in c_firewood_energy_S2_Tgr60]
HWP_S2_Tgr60_landfill = [x*1/0.82 for x in OC_storage_S2_Tgr60]

HWP_S2_Tgr60_sum = [HWP_S2_Tgr60, HWP_S2_Tgr60_energy, HWP_S2_Tgr60_landfill]
HWP_S2_Tgr60_sum = [sum(x) for x in zip(*HWP_S2_Tgr60_sum )]

## in-use stocks (S-4)
TestDSM2_Tgr60.s = [x/division for x in TestDSM2_Tgr60.s]
#TestDSM2_Tgr60.i = [x/division for x in TestDSM2_Tgr60.i]


## calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_S2_Tgr_60y = (tf,1)
stocks_S2_Tgr_60y = np.zeros(zero_matrix_stocks_S2_Tgr_60y)


i = 0
stocks_S2_Tgr_60y[0] = C_LF_S2_Tgr60[0] - Landfill_decomp_DL_FP_S2_Tgr_60y[0]

while i < tf-1:
    stocks_S2_Tgr_60y[i+1] = np.array(C_LF_S2_Tgr60[i+1] - Landfill_decomp_DL_FP_S2_Tgr_60y[i+1] + stocks_S2_Tgr_60y[i])
    i = i + 1


## calculate aggregate flow of logged wood (F1-2)
HWP_logged_S2_Tgr_60y = [x1+x2 for (x1,x2) in zip(HWP_S2_Tgr60_sum, [x*2/3 for x in c_firewood_energy_S2_Tgr60])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2_Tgr_60y = (tf,1)
ForCstocks_S2_Tgr_60y = np.zeros(zero_matrix_ForCstocks_S2_Tgr_60y)

i = 0
ForCstocks_S2_Tgr_60y[0] = initAGB - flat_list_Tgr_60y[0] - decomp_tot_S2_Tgr_60y[0] - HWP_logged_S2_Tgr_60y[0]

while i < tf-1:
    ForCstocks_S2_Tgr_60y[i+1] = np.array(ForCstocks_S2_Tgr_60y[i] - flat_list_Tgr_60y[i+1] - decomp_tot_S2_Tgr_60y[i+1] - HWP_logged_S2_Tgr_60y[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2_amount_Tgr60 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\NonRW_DLS3.xlsx', 'DLS3_45%_Tgr')
NonRW_amount_S2_Tgr_60y = df2_amount_Tgr60['NonRW_amount'].values

NonRW_amount_S2_Tgr_60y = [x/1000 for x in NonRW_amount_S2_Tgr_60y]
    


#create columns
dfM_Tgr_60y = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1c (t-C)': flat_list_Tgr_60y,
                                    'F0-1a (t-C)': emissions_refor_S2_Tgr_60y_seq,
                                    'F1-0 (t-C)': decomp_tot_S2_Tgr_60y[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Tgr_60y,
                                    #'F1c-2 (t-C)': FP_S2_Tgr_60y,
                                    'F1-2 (t-C)': HWP_logged_S2_Tgr_60y, 
                                    'St-1 (t-C)':ForCstocks_S2_Tgr_60y[:,0], 
                                    'F2-3 (t-C)': HWP_S2_Tgr60_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2_Tgr60], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2_Tgr60_sum, [x*1/0.82 for x in OC_storage_S2_Tgr60], [x*1/3 for x in c_firewood_energy_S2_Tgr60])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2_Tgr60],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2_Tgr60], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2_Tgr60.s, 
                                    #'S-4-i (t-C)': TestDSM2_Tgr60.i,
                                    'F4-5 (t-C)': TestDSM2_Tgr60.o,
                                    'F5-6 (t-C)': TestDSM2_Tgr60.o, 
                                    'F5-7 (t-C)': C_LF_S2_Tgr60,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2_Tgr60,
                                    'F6-0-2 (t-C)': TestDSM2_Tgr60.o,
                                    'St-7 (t-C)': stocks_S2_Tgr_60y[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_DL_FP_S2_Tgr_60y,
                                    'F8-0 (t-C)': PH_Emissions_HWP2_Tgr_60y,
                                    'S9-0 (t)': NonRW_amount_S2_Tgr_60y,
                                    'F9-0 (t-C)': emissions_NonRW_S2_Tgr_60y,
                                    })

##S2_E_Hbr_40y
## define the input flow for the landfill (F5-7)
OC_storage_E2_Hbr40 = dfE2_Hbr40['Other_C_storage'].values


OC_storage_E2_Hbr40 = [x/division for x in OC_storage_E2_Hbr40]
OC_storage_E2_Hbr40 = [abs(number) for number in OC_storage_E2_Hbr40]

C_LF_E2_Hbr40 = [x*1/0.82 for x in OC_storage_E2_Hbr40]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_E2_Hbr40 = [x/division for x in dfE2_Hbr40['Wood_pellets'].values]
HWP_E2_Hbr40_energy =  [x*1/3 for x in c_firewood_energy_E2_Hbr40]
HWP_E2_Hbr40_landfill = [x*1/0.82 for x in OC_storage_E2_Hbr40]

HWP_E2_Hbr40_sum = [HWP_E2_Hbr40, HWP_E2_Hbr40_energy, HWP_E2_Hbr40_landfill]
HWP_E2_Hbr40_sum = [sum(x) for x in zip(*HWP_E2_Hbr40_sum )]

## in-use stocks (S-4)
TestDSME2_Hbr40.s = [x/division for x in TestDSME2_Hbr40.s]
#TestDSM2_Tgr60.i = [x/division for x in TestDSM2_Tgr60.i]


## calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_E2_Hbr_40y = (tf,1)
stocks_E2_Hbr_40y = np.zeros(zero_matrix_stocks_E2_Hbr_40y)


i = 0
stocks_E2_Hbr_40y[0] = C_LF_E2_Hbr40[0] - Landfill_decomp_DL_FP_E2_Hbr_40y[0]

while i < tf-1:
    stocks_E2_Hbr_40y[i+1] = np.array(C_LF_E2_Hbr40[i+1] - Landfill_decomp_DL_FP_E2_Hbr_40y[i+1] + stocks_E2_Hbr_40y[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_E2_Hbr_40y = [x1+x2 for (x1,x2) in zip(HWP_E2_Hbr40_sum, [x*2/3 for x in c_firewood_energy_E2_Hbr40])] 


#calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_E2_Hbr_40y = (tf,1)
ForCstocks_E2_Hbr_40y = np.zeros(zero_matrix_ForCstocks_E2_Hbr_40y)

i = 0
ForCstocks_E2_Hbr_40y[0] = initAGB - flat_list_Hbr_40y[0] - decomp_tot_E2_Hbr_40y[0] - HWP_logged_E2_Hbr_40y[0]

while i < tf-1:
    ForCstocks_E2_Hbr_40y[i+1] = np.array(ForCstocks_E2_Hbr_40y[i] - flat_list_Hbr_40y[i+1] - decomp_tot_E2_Hbr_40y[i+1] - HWP_logged_E2_Hbr_40y[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
dfE2_amount_Hbr40 = pd.read_excel('C:\\Work\\Programming\\Paper_3_DLCA\\NonRW_DLS3.xlsx', 'DLS3_60%_Hbr')
NonRW_amount_E2_Hbr_40y = dfE2_amount_Hbr40['NonRW_amount'].values

NonRW_amount_E2_Hbr_40y = [x/1000 for x in NonRW_amount_E2_Hbr_40y]


    


#create columns
dfE2_Hbr_40y = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1c (t-C)': flat_list_Hbr_40y,
                                    'F0-1a (t-C)':emissions_refor_E2_Hbr_40y_seq,
                                    'F1-0 (t-C)': decomp_tot_E2_Hbr_40y[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Tgr_60y,
                                    #'F1c-2 (t-C)': FP_S2_Tgr_60y,
                                    'F1-2 (t-C)': HWP_logged_E2_Hbr_40y, 
                                    'St-1 (t-C)':ForCstocks_E2_Hbr_40y[:,0], 
                                    'F2-3 (t-C)': HWP_E2_Hbr40_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_E2_Hbr40], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_E2_Hbr40_sum, [x*1/0.82 for x in OC_storage_E2_Hbr40], [x*1/3 for x in c_firewood_energy_E2_Hbr40])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_E2_Hbr40],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_E2_Hbr40], 
                                    'F4-0 (t-C)': c_pellets_Hbr_40y,
                                    'St-4 (t-C)': TestDSME2_Hbr40.s, 
                                    #'S-4-i (t-C)': TestDSME2_Hbr40.i,
                                    'F4-5 (t-C)': TestDSME2_Hbr40.o,
                                    'F5-6 (t-C)': TestDSME2_Hbr40.o, 
                                    'F5-7 (t-C)': C_LF_E2_Hbr40,
                                    'F6-0-1 (t-C)': c_firewood_energy_E2_Hbr40,
                                    'F6-0-2 (t-C)': TestDSME2_Hbr40.o,
                                    'St-7 (t-C)': stocks_E2_Hbr_40y[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_DL_FP_E2_Hbr_40y,
                                    'F8-0 (t-C)': PH_Emissions_HWPE2_Hbr_40y,
                                    'S9-0 (t)': NonRW_amount_E2_Hbr_40y,
                                    'F9-0 (t-C)': emissions_NonRW_E2_Hbr_40y,
                                    })
    
    
writer = pd.ExcelWriter('C_flows_SysDef_DLS3.xlsx', engine = 'xlsxwriter')



dfM_Ac_18y.to_excel(writer, sheet_name = 'DLS3_45%_Ac_18y', header=True, index=False)
dfM_Tgr_60y.to_excel(writer, sheet_name = 'DLS3_45%_Tgr', header=True, index=False)
dfE2_Hbr_40y.to_excel(writer, sheet_name = 'DLS3_60%_Hbr', header=True, index=False)


writer.save()

#%%