# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:08:30 2022
@author: Fatemeh
various cplex parameters
"""

import os
import numpy as np 
import pandas as pd 
import sys
import psutil
import datetime
from pyomo.environ import *
import Hrcn_SCUC 

PH = Hrcn_SCUC.HurricaneSCUC('Result_ps','Images_ps')
    
Alfa = float(0.000001)
Power_system = '2K system_full.xlsx'
Load_factor = 'loadfactor.txt'
TIme0 = np.array(range(0,72)) #the hours for this network configuration

LnFlrPrblty =  'LineFailureProbability_at_Onehour.json'  #LineFailureProbability_at_Threehour
path1 = os.getcwd()
LineFailureProbability = os.path.join(path1 ,LnFlrPrblty)    

# MIP emphasis switch
# 0  Balance optimality and feasibility; default
# 1  Emphasize feasibility over optimality
# 2  Emphasize optimality over feasibility 
# 3  Emphasize moving best bound 
# 4  Emphasize finding hidden feasible solutions

# algorithm for initial MIP relaxation
# 0  Automatic: let CPLEX choose; default  
# 1  Primal Simplex 
# 2  Dual Simplex  
# 3  Network Simplex 
# 4  Barrier
# 5  Sifting
# 6  Concurrent (Dual, Barrier, and Primal)
algrthm = 6
FO = 0

item = 'AR5_'+str(algrthm)+str(FO)+'.xlsx'
item_lsh = 'LS5_'+str(algrthm)+str(FO)+'.xlsx'
 

Bus, Line, Gen, Key = PH.clean_data(Power_system) #basis bus, line and Gen data, Key is the bus numbers in the original bus data (e.g. 8230)

Load_fact_0 = np.loadtxt(Load_factor) #load factor data for 24 hours
    
LineStatus = PH.scenario_gen(Line,LineFailureProbability,Alfa) 
Load_fact = np.resize(Load_fact_0, len(LineStatus)) # extending the load factor data to the number of days in line status

LineStatus = LineStatus[TIme0].astype(int) # in case of changing the duration of hours
Load_fact = Load_fact[TIme0]

#graphs, lonely nodes, segments at each new division, time index of change in segments 
new_dict, D0, U, U_i = PH.Island(Line,LineStatus,Load_fact,Bus)

D0_abr = dict((k, v) for k, v in D0.items() if len(v) > 0) #abriviated form of lonely nodes and their hour(index) of occurance



dict_B,dict_L,dict_G,dict_g_flow,dict_S,HH0 = PH.Net_info(Bus, Line, Gen, LineStatus, Load_fact, new_dict, U, U_i)
HH2 = {}
for n in range(0,len(HH0)):
    HH2[n] = np.array(range(HH0[n,0],HH0[n,1]))

Mlines_UC = np.array([2,10,12,14,15,18,32,44,57,62,63,70,87,94,102,114,126,166,212,213,226,239,240,271,273,313,319,336,343,344,347,396,439,446,472,492,542,544,547,553,603,625,678,732,742,743,745,826,837,838,839,886,902,921,923,987,999,1008,1011,1033,1082,1133,1145,1155,1171,1202,1229,1230,1247,1250,1298,1347,1361,1400,1437,1471,1472,1477,1513,1519,1547,1600,1611,1684,1699,1704,1731,1753,1800,1806,1808,1865,1867,1929,1931,1960,2056,2059,2093,2108,2129,2195,2202,2269,2281,2312,2409,2438,2440,2533,2559,2582,2612,2630,2642,2693,2704,2727,2750,2751,2805,2836,2856,2880,2906,2914,2961,3055])  # number of lines
ind_M_UC = Mlines_UC-1 #index of lines
######################## Set the parameters ###################################
    
tolerance = 0.05 #cplex mip gap
Threads = int((psutil.cpu_count(logical=True)+psutil.cpu_count(logical=False))/2) #cplex core usage
W_mem = int(psutil.virtual_memory().total/1000000) # cplex working memory
MaxIteration = 100 #maximum number of iteration, after that the final set of monitored line will be considered as final monitored lines set.
Penalty = 1000 #penalty factor (x) for load shedding and over generation. x times more than the most expensive generation MWhr in the network
BigM = 5000 #Sufficiennt big number for cancellation transaction.
    
######################## Define Parameters ####################################
    
start_time = datetime.datetime.now()
N = len(Bus)
K = len(Line)
G = len(Gen)
T = len(Load_fact)

N1 = np.arange(N)
K1 = np.arange(K)
G1 = np.arange(G)
T1 = np.arange(T)
T2 = np.array(T1[0:-1]) #for start up
T3 = np.array(T1[1:])

LC1 = Gen[:,8]
LC2 = Gen[:,9]
LC3 = Gen[:,10]
LC4 = Gen[:,11]

######################## Flag : error in the input data #######################

#0: Pmax less than Pmin and is not positive for all generation
#1: The generation units are not located in the feasible range of bus numbers
#2: Cost index is negative
#3: LineStatus data has problem
#4: Load shedding price is zero.

Flag = np.ones(5)
if np.any(Gen[:,2] < Gen[:,3]) or np.any(Gen[:,3] < 0):
    Flag[0] = 0
    print('Check the generation data. Is the Pmax more than or equal to the Pmin and is Pmin positive for all generation units?')
    sys.exit()
if np.any(Gen[:,1] > N) or np.any(Gen[:,1] < 0):
    Flag[1] = 0
    print('Check the generation data. Are all the generation units located in the feasible range of bus numbers?')
    sys.exit()


if np.any(Gen[:,8:15] < 0):
    Flag[2] = 0
    print('Check the generation data. Negative cost!')
    sys.exit()
###adjusting the minimum up and down data: when the unit is turned on or off it should stay on or off for at least one hour
Gen[:,15] = np.where(Gen[:,15] > 0, Gen[:,15], 1)  
Gen[:,16] = np.where(Gen[:,16] > 0, Gen[:,16], 1)  
#LineStatus: line can be either online or offline
if np.isin(LineStatus,[0,1]).all():
    print('Reading data completed!')
    print('"  "')
else:
    Flag[3] = 0 #line can be either online or offline
    print('Check the LineStatus data. The line status values should be 0 or 1.')
    sys.exit()

######################## organizing the input data ############################

# FromBus = Line[:,1].astype(int)
# ToBus = Line[:,2].astype(int)
FkMax = np.zeros([T,K])
FkMin = np.zeros([T,K])
Load = np.zeros([T,N])
TotalLoad = np.zeros(T)

for t in np.arange(T):
    Load[t,:] = Bus[:,1]*Load_fact[t]
TotalLoad = np.sum(Load, axis=1)

L_bus_in = np.where(Bus[:,1]>0)[0] # index of buses with load attached to them

Loaded_bus = {} # key: index of time, value: index of bus with load in the system with outages
for i in range(0,len(Load_fact)):
      Loaded_bus[i] = np.setdiff1d(L_bus_in,(np.array(list(D0[i]))-1).astype(int))   

# Buses that are in one of the segment at t and then are isolated at t+1
Bus_RR = {}
for t in range(1,len(Load_fact)):
    Bus_RR.setdefault(t,[])
    Bus_RR[t] = np.setdiff1d(list(D0[t]), list(D0[t-1])).astype(int)

BusRR_abr = dict((k, v) for k, v in Bus_RR.items() if len(v) > 0) #time index as key and nodes that are in a seg in t and in D0 in t+1 as values 
      
# L_SH = np.zeros((len(Load_fact),len(L_bus_in)))
# OVG = np.zeros((len(Load_fact),G)) 
  
######################## Initial values and outages ###########################

NofOut = np.zeros(T,dtype = int)
NofMonitor = np.zeros(MaxIteration,dtype = int) #for the first iteration no line would be monitored
# MaxMonitor = 0

endflag = 0
COUNTER = 1

P_result = np.zeros([T,G]) #dispatch result variable
Commitment_result = np.zeros([T,G])
FC_result = np.zeros([T,K]) #results for flow cancellation
LSH_result = np.zeros([T,K]) #result for lost load
OVG_result = np.zeros([T,G]) #result for over generation


LSh_cost_bus = np.zeros(N) #load lost price for each bus(the same for all scenarios)
Over_Gen_Cost = np.zeros(G) #Over generation cost for each generation unit(the same for all scenarios)

#finding the maximum marginal generation cost    
LSh_cost_all = Penalty*np.max(np.array([LC1,LC2,LC3,LC4])) #Load Lost penalty equals to x times maximum generation cost
print('Shadow price is .......................',LSh_cost_all) 
    
#LSh_cost_all = 50, if load lost penalty needed to be a constant value
if (LSh_cost_all < 0):
    Flag[4] = 0 #Load shedding penalty can not be negative or zero
    print('Load shedding cost cannot be negative!, check the linear costs and penalty value.')
    sys.exit()

    
LSh_cost_bus[:] = LSh_cost_all
#LSh_cost_bus = LSh_cost_all #load shedding cost is considered the same value for all the buses. Any policy regarding different load lost penalties(by buses) should be applied here.
   
Over_Gen_Cost[:] = LSh_cost_all
#Over_Gen_Cost = LSh_cost_all #over generation cost is considered the same value for all the units and equal to load shedding cost. Any policy for different over generation penalties should be applied here.    

# number of outage per hour and per scenario
for t in T1:
    NofOut[t] = np.count_nonzero(LineStatus[t,:] == 0)

# MaxOut = int(np.max(NofOut))

LS_time,LS_line = np.where(LineStatus == 0)

# S: compact form of LineStatus. Arrays are line index starting from zero.   
#key: time and values: line number    
S_FC = {}
for i in dict_S.keys():
    S0 = dict_S[i]
    for t in S0.keys():
        if t in S_FC:
            S_FC[t] = np.concatenate((S_FC[t],S0[t]), axis=None) 
        else: 
            S_FC[t] = S0[t]

# The monitored lines and outages are different in each scenario and hour: FkMax and Min are defined on scenarios and hours.            
for l in K1:
    FkMax[:,l] = Line[l,5]
    FkMin[:,l] = -Line[l,5]
# if line is out, there is no need to monitor it at that hour. The original capacity multiplied by 100 should be enoug
FkMax[LS_time,LS_line] = FkMax[LS_time,LS_line]*100
FkMin[LS_time,LS_line] = FkMin[LS_time,LS_line]*100  

linemonitorflag = np.zeros((T,K)).astype(int) #0: not monitored, 1:monitored (for the first iteration no line would be monitored.)
# linemonitorflag_all = np.zeros((10*T,K)).astype(int)

#M is the compact form of monitored lines. Arrays are line numbers starting from zero.
M = {} #np.zeros(K,dtype=int)

Done_b4 = {} # seg: {m,t}
for sg in dict_L.keys():
    Done_b4.setdefault(sg,[])  
    M4 = {}
    for l in (dict_L[sg][:,0]-1).astype(int):
        M4.setdefault(l,[])   
    Done_b4[sg] = M4
    
#print("Shift factor matrix calculating...")
shift1_time = datetime.datetime.now()

Dct_shiftfactor = PH.PTDF(Bus, Line, new_dict, U, U_i)
shift2_time = datetime.datetime.now()
print("Shift_factor calculated: ",(shift2_time-shift1_time).total_seconds())
main_solve = datetime.datetime.now()
print('Started solve at: ',main_solve)

######################## Iterative Optimization ###############################
    
print("******------------------------******------------------------******")       
print("Defining Optimization Problem...")

model = ConcreteModel()

## Set Variables
def ub_pmax(model, t,g):
    return (0, Gen[g,2])
def ub_p1(model, t,g):
    return (0, Gen[g,4])
def ub_p2(model, t,g):
    return (0, Gen[g,5])
def ub_p3(model, t,g):
    return (0, Gen[g,6])
def ub_p4(model, t,g):
    return (0, Gen[g,7])
def ub_LSH(model, t,b):
    return (0, Load[t,b])
def ul_FK(model, t,m):
    return (FkMin[t,m],FkMax[t,m])

model.P = Var(T1,G1, bounds = ub_pmax) #Generation dispatch (Total)(6.0) different for scenarios
model.P1 = Var(T1,G1, bounds = ub_p1) #Generation dispatch (1st segment)(6.0) different for scenarios
model.P2 = Var(T1,G1, bounds = ub_p2) #Generation dispatch (2nd segment)(6.0) different for scenarios
model.P3 = Var(T1,G1, bounds = ub_p3) #Generation dispatch (3rd segment)(6.0) different for scenarios
model.P4 = Var(T1,G1, bounds = ub_p4) #Generation dispatch (4th segment)(6.0) different for scenarios
model.Fk = Var(T1,K1, bounds = ul_FK) #model.fk_set
model.uk = Var(T1,G1, within = Binary) #Generator status (1:on, 0:off) (6.0) same for all scenarios
model.v = Var(T2,G1, within = Binary) #Startup variable (1:startup) (6.0) same for all scenarios
model.w = Var(T2,G1, within = Binary) #Shutdown variable (1:shutdown) (6.0) same for all scenarios
model.FC = Var(T1,K1, bounds = (-BigM,BigM)) # model.fc_set #Flow cancellation variable (6.0) different for scenarios
model.LSH = Var(T1,N1, bounds = ub_LSH) #model.lsh_set #Load shedding (lost load) variable (6.0) different for scenarios
model.OVG = Var(T1,G1, bounds = ub_pmax) #Overgeneration variable (6.0) different for scenarios
model.flow = Block() 
model.flow.cncl = ConstraintList() #line monitor constraint

for t in D0_abr.keys():
    for n in D0_abr[t]:
        G_D0 = np.where(Gen[:,1]==n)[0]
        for g in G_D0:
            model.P[t,g].fix(0)
            model.uk[t,g].fix(0)
        if n-1 in L_bus_in: # n: bus number, L_bus_in: bus index
            n1 = int(n-1)
            model.LSH[t,n1].fix(Load[t,n1])
            
# Define the objective function
def obj_rule(model):
    return sum(model.P1[t,g]*LC1[g]+model.P2[t,g]*LC2[g]+model.P3[t,g]*LC3[g]+model.P4[t,g]*LC4[g]+Over_Gen_Cost[g]*model.OVG[t,g] for t in T1 for g in G1)+sum(LSh_cost_bus[b]*model.LSH[t,b] for t in T1 for b in L_bus_in)+sum(model.v[t,g]*Gen[g,12]+ model.w[t,g]*Gen[g,13] for t in T2 for g in G1)+sum(model.uk[t,g]*Gen[g,14] for t in T1 for g in G1)                
model.obj = Objective(rule=obj_rule)

def Ptotal_rule(model,t,g):
      return model.P[t,g] == model.P1[t,g]+model.P2[t,g]+model.P3[t,g]+model.P4[t,g]
model.Ptotal = Constraint(T1,G1,rule=Ptotal_rule)


model.node_balance_rule = ConstraintList()
print('Adding network power balance constraints...')
for sg in dict_L.keys():
    Bus_seg = dict_B[sg] 
    Gen_seg = dict_G[sg]
    
    N_seg = (Bus_seg[:,0]-1).astype(int) 
    G_sg = (Gen_seg[:,0]-1).astype(int)
       
    h = int(str(sg).split(",",1)[0])
    Load_sg = np.zeros([T,N])
    TotalLoad_sg = np.zeros(T)
    for t in HH2[h]:  
        for ix,b in enumerate(N_seg): 
            Load_sg[t,b] = Bus_seg[ix,1]*Load_fact[t]
    TotalLoad_sg = np.sum(Load_sg, axis=1)
    for t in HH2[h]:
        model.node_balance_rule.add(sum(model.P[t,g]-model.OVG[t,g] for g in G_sg) + sum(model.LSH[t,b] for b in N_seg)==TotalLoad_sg[t])
        
# def node_balance_rule(model,t):
#     return sum(model.P[t,g]-model.OVG[t,g] for g in G1) + sum(model.LSH[t,b] for b in L_bus_in)==TotalLoad[t] 
# model.node_balance = Constraint(T1,rule=node_balance_rule) 

print('Adding commitment constraints...')
def Ineq9_rule(model,t,g):
    return model.P[t,g]-model.uk[t,g]*Gen[g,2]<=0 
model.Ineq9 = Constraint(T1,G1,rule=Ineq9_rule)

def Ineq10_rule(model,t,g): 
    return model.uk[t,g]*Gen[g,3]-model.P[t,g]<=0 
model.Ineq10 = Constraint(T1,G1,rule=Ineq10_rule)

#start up and shut down variables
print('Adding start-up and shut down constraints...')  
def vw1_rule(model,t,g):
    return model.v[t-1,g]-model.w[t-1,g]-model.uk[t,g]+model.uk[t-1,g]==0    
model.vw1 = Constraint(T3,G1,rule=vw1_rule)

def vw2_rule(model,t,g):
    return model.v[t-1,g]+model.w[t-1,g] <= 1
model.vw2 = Constraint(T3,G1,rule=vw2_rule)

print('Adding ramping constraints ...')
def ramp_rule(model,t,i5):
    if len(BusRR_abr.keys())>0:
        if t in BusRR_abr.keys():
            if Gen[i5,1] in BusRR_abr[t]:
                return (-Gen[i5,2],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,2])
            else:
                return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
        else:
            return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
    else:
        return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
if T >1:
    model.ramp = Constraint(T3,np.where(Gen[:,17] < Gen[:,2])[0],rule=ramp_rule)

model.minU = ConstraintList()
model.minD = ConstraintList()
model.flow_can  = ConstraintList()


#Inequality constraints
        #minimum up and down time and ramping contraints exist only when t > 1
if T > 1:
    #adding minimum up and down time constraint
    print('Adding min up/down time constraints ...')
    for i6 in np.where(Gen[:,16]>1)[0]:
        for m in range(1, int(T+1 - Gen[i6,16])): # minimum up time constraints in the middle
            model.minU.add(sum(model.uk[i,i6] for i in range(m,int(m+Gen[i6,16])))-Gen[i6,16]*(model.uk[m,i6]-model.uk[m-1,i6])>=0)
        for m in range(1, int(T+1 -Gen[i6,15])):
            model.minD.add(Gen[i6,15] - sum(model.uk[i,i6] for i in range(m,int(m+Gen[i6,15])))-Gen[i6,15]*(model.uk[m-1,i6]-model.uk[m,i6])>=0) 

print('Adding line flow constraints ...')
line_con_s0 = datetime.datetime.now()

for sg in dict_L.keys():
    Bus_seg = (dict_B[sg][:,0]).astype(int)
    shiftfactor = Dct_shiftfactor[sg]
    Line_seg = dict_L[sg]
    
    Dict_g = dict_g_flow[sg]
    h = int(str(sg).split(",",1)[0])
    for m in ind_M_UC:
        if m+1 in Line_seg[:,0]:
            ind_m = int(np.where(m==Line_seg[:,0]-1)[0])
            linemonitorflag[HH2[h],m] = 1
            HH1 = Done_b4[sg][m]
            for t in filter(lambda el: el not in HH1, HH2[h]):
                LF = 0
                for b in range(1,len(Bus_seg)):
                    if Bus_seg[b]-1 in Loaded_bus[t]:
                        LF += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                    if Bus_seg[b] in Dict_g.keys():                    
                        LF += sum(shiftfactor[ind_m,b-1]*(model.P[t,i2]-model.OVG[t,i2]) for i2 in Dict_g[Bus_seg[b]])
                if sg in dict_S.keys():
                    S = dict_S[sg]
                    if t in S.keys():
                        for o in range(0,len(S[t])):
                            ind_mo = np.where(S[t][o]==Line_seg[:,0])[0]
                            ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg[:])[0])
                            ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg[:])[0])
                            if ind_FB == 0:
                                LF += -shiftfactor[ind_m,ind_TB-1]*model.FC[t,S[t][o]-1]
                            elif ind_TB == 0:
                                LF += shiftfactor[ind_m,ind_FB-1]*model.FC[t,S[t][o]-1]
                            else:
                                LF += (shiftfactor[ind_m,ind_FB-1]-shiftfactor[ind_m,ind_TB-1])*model.FC[t,S[t][o]-1]
                    
                model.flow.cncl.add(LF == model.Fk[t,m])  
                Done_b4[sg][m].append(t)
line_con_f0 = datetime.datetime.now()
print('Line monitoring constraints take: ',(line_con_f0-line_con_s0).total_seconds()) 

print('Adding line cancelation constraints ...')
line_con_f1 = datetime.datetime.now()
for i in dict_S.keys(): #index of time
    Bus_seg = (dict_B[i][:,0]).astype(int)
    shiftfactor = Dct_shiftfactor[i]
    Line_seg = dict_L[i]
    S = dict_S[i]
    Dict_g = dict_g_flow[i] #key:number of bus with gen, value: index of gen
    
    for t in S.keys():
        for o in range(0,len(S[t])):
            VT = 0
            ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
            for b in range(1,len(Bus_seg)):
                if Bus_seg[b]-1 in Loaded_bus[t]:
                    VT += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                if Bus_seg[b] in Dict_g.keys():
                    VT += sum(shiftfactor[ind_m,b-1]*(model.P[t,i1]-model.OVG[t,i1]) for i1 in Dict_g[Bus_seg[b]]) 
            VT += -model.FC[t,S[t][o]-1] #second part of eq 25
            for oo in range(0,len(S[t])):
                ind_mo = int(np.where(S[t][oo]==Line_seg[:,0])[0])
                ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg)[0])
                ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg)[0])
                if ind_FB == 0:
                    VT += -shiftfactor[ind_m,ind_TB-1]*model.FC[t,S[t][oo]-1]
                elif ind_TB == 0:
                    VT += shiftfactor[ind_m,ind_FB-1]*model.FC[t,S[t][oo]-1]
                else:
                    VT += (shiftfactor[ind_m,ind_FB-1]-shiftfactor[ind_m,ind_TB-1])*model.FC[t,S[t][oo]-1]
            model.flow_can.add(VT == 0)
            #model.flow_cncl.pprint()

line_con_f = datetime.datetime.now()
print('Line outage constraints take: ',(line_con_f-line_con_f1).total_seconds())
print('Constraints, Done!')


opt = SolverFactory('cplex_persistent',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex')
opt.set_instance(model)    
opt.options['mip_tolerances_mipgap'] = tolerance #to instruct CPLEX fot stop as soon as it has found a feasible integer solution proved to be within five percent of optimal 
opt.options['threads'] = Threads
opt.options['workmem'] = W_mem  
opt.options['mip_strategy_startalgorithm'] = algrthm
opt.options['emphasis_mip'] = FO # feasibility

Solution=opt.solve(model,tee=True,report_timing=True)
Solution.write(num=1)

print("Construction of persistent model complete.")
def_f_p = datetime.datetime.now()
print('construction time', (def_f_p - start_time).total_seconds())


# res_s = datetime.datetime.now()
for t in T1:
    for g in G1:
        P_result[t,g] = model.P[t,g].value
        OVG_result[t,g] = model.OVG[t,g].value
        Commitment_result[t,g] = model.uk[t,g].value
    for b in L_bus_in:
        LSH_result[t,b] = model.LSH[t,b].value
for t in S_FC.keys():
    for o in S_FC[t]:
        FC_result[t,o-1] = model.FC[t,o-1].value

######################## LINE MONITOR DECISION ################################

print('Calculating line flows ...')
load = np.zeros([N-1,T])
gen = np.zeros([N-1,T])
endflag = 0

L_bus_in_no0 = np.delete(L_bus_in,np.where(L_bus_in==0)[0])
for t in T1:
    for b in L_bus_in_no0:
        load[b-1,t] = Load[t,b]
        gen[b-1,t] = LSH_result[t,b] #the effect of load shedding is equal to the same amount of generation at that bus
    for i4 in np.where(Gen[:,1] != 1)[0]:
        gen[int(Gen[i4,1]-2),t] += P_result[t,i4]-OVG_result[t,i4]
                
line_flow_load_tmp = np.zeros((K,T))
line_flow_gen_tmp = np.zeros((K,T))
for h in Dct_shiftfactor.keys():
    t = int(str(h).split(",",1)[0])
    L_seg = (dict_L[h][:,0]-1).astype(int) # line_flow_load_tmp of lines L_seg is to be calculated
    B_seg = (dict_B[h][:-1,0]-1).astype(int) # load and gen were created for b in range(1,N)
    shiftfactor = Dct_shiftfactor[h]
    line_flow_load_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,load[B_seg,HH0[t,0]:HH0[t,1]]) 
    line_flow_gen_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,gen[B_seg,HH0[t,0]:HH0[t,1]])
       
        

line_flow_load = np.transpose(line_flow_load_tmp)
line_flow_gen = np.transpose(line_flow_gen_tmp)
line_flow = line_flow_gen-line_flow_load

for i in dict_S.keys():
    Bus_seg = dict_B[i]
    shiftfactor = Dct_shiftfactor[i]
    Line_seg = dict_L[i]
    S = dict_S[i]
    
    for t in S.keys():
        for l in range(0,len(Line_seg)):    
            for o in range(0,len(S[t])):
                ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
                ind_TB = int(np.where(int(Line_seg[ind_m,2])==Bus_seg[:,0])[0])
                ind_FB = int(np.where(int(Line_seg[ind_m,1])==Bus_seg[:,0])[0])
                if ind_FB == 0:
                    line_flow[t,int(Line_seg[l,0]-1)] += -shiftfactor[l,ind_TB-1]*FC_result[t,S[t][o]-1]
                elif ind_TB == 0:
                    line_flow[t,int(Line_seg[l,0]-1)] += shiftfactor[l,ind_FB-1]*FC_result[t,S[t][o]-1]
                else:
                    line_flow[t,int(Line_seg[l,0]-1)] += (shiftfactor[l,ind_FB-1]-shiftfactor[l,ind_TB-1])*FC_result[t,S[t][o]-1]
    
M = {}   
for t in T1:
    M.setdefault(t,[])
    
for t,l in zip(*np.where(linemonitorflag == 0)):
    if (line_flow[t,l]>FkMax[t,l]) or (line_flow[t,l]<FkMin[t,l]):
        M[t].append(l)
        linemonitorflag[t,l] = 1
        endflag = 0

M = dict((k, v) for k, v in M.items() if len(v) > 0)

print('Previous number of monitored instances = ',NofMonitor[0]) 

    
NofMonitor[COUNTER] = len(np.where(linemonitorflag==1)[0]) 
print('number of monitored instances = ',NofMonitor[COUNTER])  
        
if len(M) == 0:
    endflag = 1
        
while endflag == 0:  
    COUNTER += 1    
    print("------------------- iteration -------------------",COUNTER)          

    #add DC power flow constraints
    # eq 24
    print('Adding line flow constraints ...')
    line_con_s = datetime.datetime.now()
    
    M2  = {}   
    M2 = PH.Segment_finder(HH2,dict_L,M)  # seg:{m:t}
    for sg in M2.keys():
        Bus_seg = (dict_B[sg][:,0]).astype(int)
        shiftfactor = Dct_shiftfactor[sg]
        Line_seg = dict_L[sg]
        
        Dict_g = dict_g_flow[sg]
        h = int(str(sg).split(",",1)[0])
        for m in M2[sg].keys():
            ind_m = int(np.where(m==Line_seg[:,0]-1)[0])

            linemonitorflag[HH2[h],m] = 1
            HH1 = Done_b4[sg][m]
            for t in filter(lambda el: el not in HH1, HH2[h]):
                LF = 0
                for b in range(1,len(Bus_seg)):
                    if Bus_seg[b]-1 in Loaded_bus[t]:
                        LF += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                    if Bus_seg[b] in Dict_g.keys():                    
                        LF += sum(shiftfactor[ind_m,b-1]*(model.P[t,i2]-model.OVG[t,i2]) for i2 in Dict_g[Bus_seg[b]])
                if sg in dict_S.keys():
                    S = dict_S[sg]
                    if t in S.keys():
                        for o in range(0,len(S[t])):
                            ind_mo = np.where(S[t][o]==Line_seg[:,0])[0]
                            ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg[:])[0])
                            ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg[:])[0])
                            if ind_FB == 0:
                                LF += -shiftfactor[ind_m,ind_TB-1]*model.FC[t,S[t][o]-1]
                            elif ind_TB == 0:
                                LF += shiftfactor[ind_m,ind_FB-1]*model.FC[t,S[t][o]-1]
                            else:
                                LF += (shiftfactor[ind_m,ind_FB-1]-shiftfactor[ind_m,ind_TB-1])*model.FC[t,S[t][o]-1]
                    
                model.flow.cncl.add(LF == model.Fk[t,m])  
                Done_b4[sg][m].append(t)
                    
                    
    line_con_f1 = datetime.datetime.now()
    print('Line monitoring constraints take: ',(line_con_f1-line_con_s).total_seconds()) 
    
    opt = SolverFactory('cplex_persistent',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex')
    opt.set_instance(model)
    opt.options['mip_tolerances_mipgap'] = tolerance #to instruct CPLEX fot stop as soon as it has found a feasible integer solution proved to be within five percent of optimal 
    opt.options['threads'] = Threads
    opt.options['workmem'] = W_mem  
    opt.options['mip_strategy_startalgorithm'] = algrthm
    opt.options['emphasis_mip'] = FO # feasibility
    
    opt.solve(model,tee=True,report_timing=True) #model,save_results=False  
    
    Solution=opt.solve(model,tee=True,report_timing=True)
    Solution.write(num=1)
    
    print(value(model.obj))
    res_s1 = datetime.datetime.now()
    for t in T1:
        for g in G1:
            P_result[t,g] = model.P[t,g].value
            OVG_result[t,g] = model.OVG[t,g].value
            Commitment_result[t,g] = model.uk[t,g].value
        for b in L_bus_in:
            LSH_result[t,b] = model.LSH[t,b].value
    for t in S_FC.keys():
        for o in S_FC[t]:
            FC_result[t,o-1] = model.FC[t,o-1].value

######################## LINE MONITOR DECISION ################################

    print('Calculating line flows ...')
    load = np.zeros([N-1,T])
    gen = np.zeros([N-1,T])
    endflag = 0

    for t in T1:
        for b in L_bus_in_no0:
            load[b-1,t] = Load[t,b]
            gen[b-1,t] = LSH_result[t,b] #the effect of load shedding is equal to the same amount of generation at that bus
        for i4 in np.where(Gen[:,1] != 1)[0]:
            gen[int(Gen[i4,1]-2),t] += P_result[t,i4]-OVG_result[t,i4]

                    
    line_flow_load_tmp = np.zeros((K,T))
    line_flow_gen_tmp = np.zeros((K,T))
    for h in Dct_shiftfactor.keys():
        t = int(str(h).split(",",1)[0])
        L_seg = (dict_L[h][:,0]-1).astype(int) # line_flow_load_tmp of lines L_seg is to be calculated
        B_seg = (dict_B[h][:-1,0]-1).astype(int) # load and gen were created for b in range(1,N)
        shiftfactor = Dct_shiftfactor[h]
        line_flow_load_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,load[B_seg,HH0[t,0]:HH0[t,1]]) 
        line_flow_gen_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,gen[B_seg,HH0[t,0]:HH0[t,1]])
           
            
    
    line_flow_load = np.transpose(line_flow_load_tmp)
    line_flow_gen = np.transpose(line_flow_gen_tmp)
    line_flow = line_flow_gen-line_flow_load

    for i in dict_S.keys():
        Bus_seg = dict_B[i]
        shiftfactor = Dct_shiftfactor[i]
        Line_seg = dict_L[i]
        S = dict_S[i]
        
        for t in S.keys():
            for l in range(0,len(Line_seg)):    
                for o in range(0,len(S[t])):
                    ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
                    ind_TB = int(np.where(int(Line_seg[ind_m,2])==Bus_seg[:,0])[0])
                    ind_FB = int(np.where(int(Line_seg[ind_m,1])==Bus_seg[:,0])[0])
                    if ind_FB == 0:
                        line_flow[t,int(Line_seg[l,0]-1)] += -shiftfactor[l,ind_TB-1]*FC_result[t,S[t][o]-1]
                    elif ind_TB == 0:
                        line_flow[t,int(Line_seg[l,0]-1)] += shiftfactor[l,ind_FB-1]*FC_result[t,S[t][o]-1]
                    else:
                        line_flow[t,int(Line_seg[l,0]-1)] += (shiftfactor[l,ind_FB-1]-shiftfactor[l,ind_TB-1])*FC_result[t,S[t][o]-1]
                    
    M = {}   
    for t in T1:
        M.setdefault(t,[])
        
    for t,l in zip(*np.where(linemonitorflag == 0)):
        if (line_flow[t,l]>FkMax[t,l]) or (line_flow[t,l]<FkMin[t,l]):
            M[t].append(l)
            linemonitorflag[t,l] = 1
            endflag = 0

    M = dict((k, v) for k, v in M.items() if len(v) > 0)  
    
    print('Previous number of monitored instances = ',NofMonitor[COUNTER-1])
    NofMonitor[COUNTER] = len(np.where(linemonitorflag==1)[0]) #np.where(linemonitorflag==1)[0],np.unique(np.concatenate(list(M.values())))
    print('number of monitored instances = ',NofMonitor[COUNTER])
    
    res_f1 = datetime.datetime.now()
    print('Calculating flows takes: ',(res_f1-res_s1).total_seconds())
    print('Counter {} finished at: {}'.format(COUNTER, res_f1)) 
    
    if len(M)>0:
        endflag = 0

    else:
        endflag = 1

    if COUNTER == (MaxIteration-1):
        endflag = 1
                   
    if endflag == 1:
        break
print('All Done!')
print('Number of Iteration: ',COUNTER)
print('Started at: ',start_time)
mainfinish = datetime.datetime.now()
print('Finished at: ',mainfinish) 
print((mainfinish-start_time).total_seconds())

Cost = value(model.obj)

# find what new nodes get seperated at each hour               
LSH_b = np.where(sum(LSH_result[t,:] for t in T1) > 0)[0]
if len(LSH_b)>0:
    LSH = np.zeros((len(LSH_b),T+2))
    for i in range(0,len(LSH_b)):
        LSH[i,0] = Key[int(LSH_b[i]),1]
        LSH[i,1] = sum(LSH_result[t,int(LSH_b[i])] for t in T1)  #the first column is the numbering used by power group(1 to number of buses)#second column is the indexing used by civil group (row position in data) and the third column is the real bus numbers
        LSH[i,2:] = LSH_result[:,int(LSH_b[i])]
else: 
    LSH = []
        
                
Summery = [{'Solution Value': value(model.obj),
            'Total Time':(mainfinish-start_time).total_seconds(),
            'Load Lost Penalty': LSh_cost_all,
            'Total Over Generation':np.sum(OVG_result),
            'Total Load shedding':np.sum(LSH_result),
            'Number of Iteration': COUNTER}]
np.set_printoptions(threshold=np.inf)
df_Lsh= pd.DataFrame(LSH)
df_cmtmnt = pd.DataFrame(Commitment_result)
df_Mnt_l = pd.DataFrame(np.unique(np.where(linemonitorflag == 1)[1]))
df_Mnt_ln = df_Mnt_l+1
df_gen = pd.DataFrame(P_result)
df_smry = pd.DataFrame(Summery)

path = os.getcwd()
folder_name = 'Result_ps'
path = os.path.join(path, folder_name)
os.makedirs(path, exist_ok=True)
item_n = os.path.join(path, item)
writer = pd.ExcelWriter(item_n) #,item
df_smry.to_excel(writer,sheet_name='Summery',index=False)
df_cmtmnt.columns =[i for i in Gen[:,0].astype(int)] 
df_cmtmnt.index=[i for i in range(1,len(Load_fact)+1)]
df_cmtmnt.to_excel(writer,sheet_name='Commitment') 
df_Mnt_ln.index=[i for i in range(1,len(df_Mnt_l)+1)]
df_Mnt_ln.to_excel(writer,sheet_name='Monitor')
df_gen.columns =[i for i in Gen[:,0].astype(int)] 
df_gen.index=[i for i in range(1,len(Load_fact)+1)]
df_gen.to_excel(writer,sheet_name='Ptotal')
#df_Lsh.columns =[i for i in range(0,T+1)] 
df_Lsh.to_excel(writer,sheet_name='Lshedding')

writer.save()
    

xlsload_t = pd.ExcelFile('Inf_GIS.xlsx')
    
Bus_loc_0 = pd.read_excel(xlsload_t, 'Buses',header=1).fillna(0)
Bus_loc_1 = Bus_loc_0[['Substation Longitude', 'Substation Latitude', 'Number']].to_numpy()

Bus_loc_2 = np.zeros((len(LSH),4+len(Load_fact)))
for i in range(0,len(LSH)):
    Bus_loc_2[i,1:3] = Bus_loc_1[np.where(LSH[i,0]==Bus_loc_1[:,2])[0],0:2]
Bus_loc_2[:,0] = LSH[:,0] #Bus number
Bus_loc_2[:,3] = Bus[LSH_b,-1] #Zone number

Bus_loc_2[:,4:] = LSH[:,2:]

Bus_loc_2[Bus_loc_2==0] = 'NaN'

clmn_B1 = list(range(1,1+len(Load_fact)))
clmn_B = ['Bus','Longt','Lat','Zone']+clmn_B1
    
np.set_printoptions(threshold=np.inf)
Bus_loc_2_final= pd.DataFrame(Bus_loc_2)
item_lsh = os.path.join(path, item_lsh)
writer = pd.ExcelWriter(item_lsh)
Bus_loc_2_final.columns = clmn_B
Bus_loc_2_final.index=[i for i in range(1,len(LSH)+1)]
Bus_loc_2_final.to_excel(writer,sheet_name='Lshedding_loc')
writer.save()    
    
# PH.vis_img(item_lsh)