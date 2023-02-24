
import os
import numpy as np 
import pandas as pd 
import json
import sys
import networkx as nx
import psutil
import datetime
import time
from pyomo.environ import *
import cv2
import plotly.express as px
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

class HurricaneSCUC:   
    
    def __init__(self, out_f,out_im_f,pvideo=True):
        self.out_f = out_f  # output folder
        self.out_im_f = out_im_f  # output folder
    def clean_data(self,data0):
        xlsload = pd.ExcelFile(data0)
        
        Bus0 = pd.read_excel(xlsload, 'Bus',header=1).sort_values('Number',ignore_index=True).fillna(0)
        Bus1 = Bus0[['Number','Load MW','Gen MW','Zone Num']].to_numpy()
        Bus = np.zeros((len(Bus0),4))
        Key_org2sorted = np.zeros((len(Bus0),2))
        Bus[:,0] = np.arange(len(Bus0))+1
        Bus[:,1:] = Bus1[:,1:]
        Key_org2sorted[:,0] = Bus[:,0]
        Key_org2sorted[:,1] = Bus1[:,0]
        
        Gen0 = pd.read_excel(xlsload, 'Gen',header=1).fillna(0)
    
        Gen1 = Gen0[['Number of Bus','Max MW','Min MW','Gen MW','MWh Price 1','MWh Price 2','MWh Price 3','MWh Price 4','Fixed Cost($/hr)','Start Up Cost','Shut Down Cost','Ramp Rate','Min Up Time','Min Down Time']].to_numpy()
        #                  0             1        2        3           4              5           6               7             8                 9              10             11           12              13
        Gen = np.zeros((len(Gen1),18))
        # [gen num,bus number,pmax,pmin,v1,v2,v3,v4,cost1,cost2,cost3,cost4,startup cost,shutdown cost,no load cost,min up,min down,ramp rate]
        # [0,          1,       2,   3,  4, 5, 6, 7,  8,   9,     10,   11 ,    12,         13,           14,         15,     16,     17     ] 
        Gen[:,0] = np.arange(len(Gen1))+1 # gen number
        for i in range(0,len(Gen1)):
            Gen[i,1] = int(np.where(Gen1[i,0] == Key_org2sorted[:,1])[0]+1) #bus number
        Gen[:,2] = Gen1[:,1]#pmax
        Gen[:,3] = Gen1[:,2]#pmin
        Gen[:,4] = Gen1[:,2] #v1
        Gen[:,5] = (Gen1[:,1] - Gen1[:,2])/3 #v2     
        Gen[:,6] = (Gen1[:,1] - Gen1[:,2])/3 #v3
        Gen[:,7] = (Gen1[:,1] - Gen1[:,2])/3 #v4
        Gen[:,8] = Gen1[:,4] #cost1     
        Gen[:,9] = Gen1[:,5] #cost2
        Gen[:,10] = Gen1[:,6] #cost3  
        Gen[:,11] = Gen1[:,7] #cost4  
        Gen[:,12] = Gen1[:,9] #start up cost 
        Gen[:,13] = Gen1[:,10] #shut down cost
        Gen[:,14] = Gen1[:,8] #noload cost  
        Gen[:,15:17] = Gen1[:,12:14] #min up and down time
        Gen[:,17] = Gen1[:,11] #ramp rate
        
        Line0 = pd.read_excel(xlsload, 'Branch',header=1).fillna(0)
        Line1 = Line0[['From Number','To Number','B','X','Lim MVA A']].to_numpy()
        #                  0             1        2   3      4           
        #data from civil engineering
        
        Brnch0 = pd.read_excel(xlsload, 'Branch state',header=1).fillna(0)
        Brnch1 = Brnch0[['From Number','To Number']].to_numpy()
        
        Brnch2 = np.zeros((len(Brnch1),6))  # [Line num,from bus,to bus,B,X,thermal limit]
        Brnch2[:,0] = np.arange(len(Brnch1))+1 # Line num
        Brnch2[:,1:3] = Brnch1  # from bus,to bus
        #,B,X,thermal limit
        for i in range(0,len(Brnch1)):
            tmp_b = np.where((Brnch1[i,0]==Line1[:,0])&(Brnch1[i,1]==Line1[:,1]))[0][0]
            Brnch2[i,[3,5]] = Line1[tmp_b,[2,4]]
            Brnch2[i,4] = -1/Line1[tmp_b,3]
        
        # [Line num,from bus,to bus,B,X,thermal limit]
        # [   0,      1,       2,   3,4,    5] 
        #Line[:,0] = np.arange(len(Line1))+1 # line number
        Line = np.zeros((len(Line1),6)) 
        Line[:,[0,3,4,5]] = Brnch2[:,[0,3,4,5]]
        for i in range(0,len(Brnch2)):
            Line[i,1] = int(np.where(Brnch2[i,1] == Key_org2sorted[:,1])[0]+1) #from bus number
            Line[i,2] = int(np.where(Brnch2[i,2] == Key_org2sorted[:,1])[0]+1) #to bus number    
            
            
        return Bus, Line, Gen, Key_org2sorted

    def ReadJSN(self,Datafile):
        # just the line numbers out of the lines_failure file
        # import json
        # Datafile = 'T201708221800.json'
        with open(Datafile, 'r') as f:
            d1 = json.load(f)

        r1 = list(d1.keys()) #convert the dictionary keys to list 
        r2 = [] #output
        for i in range(0,len(r1)):
            iy = int(r1[i]) #intermediary
            r2.append(iy)
        
        Ot1 = np.array(r2) #output1: the line numbers as int     
            #the time stamps under study
        ts = np.array(list(d1[r1[0]].keys()))# the time stamps are the same for all of the lines: keys of the first key of original dictionary 

        #the line number as the first entry in the numpy array and the 1:18 are the probability of line outage for the timesteps under study:
        Otg_prb = np.zeros([len(r2),len(ts)+1])
        for i in range(0,len(r2)):
            Otg_prb[i,0] = Ot1[i]
            Otg_prb[i,1:] = list(d1[r1[i]].values())
            
        return Otg_prb,ts


    def try_parsing_date(self,text):
        for fmt in ("'%m_%d_%H%M'", '%m_%d_%H%M'):
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise ValueError('no valid date format found')    

    #data0:Line,data1:Lines_failure probability,data2:time stamps of hurricane probability data,data3:alfa
    def scenario_gen(self, data0,data1,data2='0.9'):

        Alfa = float(data2) #Probability threshold for line to be considered out
        #the first column: line number, the rest of columns the outage probability of line at various time steps
        #input data from civil E group for line failure probability
        
        ln_flr_prb,fin = self.ReadJSN(data1) #'Lines_failure.json', time steps


        time_s = np.zeros((len(fin),3),dtype=int)
        for i in range(0,len(fin)):
            datetime_object = self.try_parsing_date(fin[i])
            time_s[i,0] = datetime_object.month 
            time_s[i,1] = datetime_object.day 
            time_s[i,2] = datetime_object.hour
        if len(fin)>1:
            smltn_dur = 24*((self.try_parsing_date(fin[-1])-self.try_parsing_date(fin[0])).days+1)
        hrrcn_hour = np.zeros(len(time_s),dtype=int) #hours of hurricane probability as time series: 12, 18, 21, 0=24, 3=27, ...
        tm0 = np.unique(time_s[:,1]) # temperary variable for the days of hurricane to help with days in different months  
        for i in range(0,len(time_s)):
            for j in range(0,len(tm0)):
                if time_s[i,1] == tm0[j]:
                    hrrcn_hour[i] = 24*j+time_s[i,2]

        LnSts = np.ones([int(smltn_dur),len(data0)]) #LineStatus

        for l in range(0,len(ln_flr_prb)):
            for t in range(0,len(time_s)): 
                if ln_flr_prb[l,t+1] >= np.abs(Alfa): #if by mistake the alfa values are entered as negative values
                    k1 = int(ln_flr_prb[l,0])#line number
                    k2 = hrrcn_hour[t]
                    LnSts[k2:,k1] = 0  
                    #break                
        return LnSts

    def Island(self,data0,data1,data2,data3):#line data, line_status, Load_factor, bus data
        #data0,data1,data2,data3 = Line,LineStatus,Load_fact,Bus
        T_isl = len(data2) #time duration in which islanding is examined
        num_bus = len(data3) #number of lines of the network
        new_dict = dict.fromkeys([i for i in range(0,T_isl)]) #Graphs are saved as dictionaries, keys: hours starting from 0 
        Base_graph = nx.Graph([(data0[j,1],data0[j,2]) for j in range(0,len(data0))]) #for j in range(0,len(C))
        #for each hour create the network of nodes and edges (bus and line)
        D0 = dict.fromkeys([i for i in range(0,T_isl)]) # graph informations: number of segments formed
        A_temp = np.zeros((T_isl)) 
        
        for i in range(0,T_isl):                
            C = np.delete(data0,np.where(data1[i,:]<1)[0],axis=0)  #line outages as data1 contains only 0 and 1
            new_dict[i] = nx.Graph([(C[j,1],C[j,2]) for j in range(0,len(C))]) 
            D0[i] = Base_graph.nodes-new_dict[i].nodes  # the lone nodes #new_dict[0]          
        for i in range(0,T_isl):   
            A_temp[i] = len(list(nx.connected_components(new_dict[i])))
            
        A2 = A_temp.astype(int)
        temp = np.max(A_temp).astype(int)
        A1 = np.zeros((T_isl,temp))
        for i in range(0,T_isl):
            A3 = A2[i]
            for j in range(0,A3):
                A1[i,j] = len(list(nx.connected_components(new_dict[i]))[j])
        a_idx = np.argsort(-A1)
        A4 = np.take_along_axis(A1, a_idx, axis=1)
        
        U0,U_ind = np.unique(A4,axis=0,return_index=True)
        U_i = np.flip(U_ind) 
        U = np.flipud(U0)
          
        return new_dict,D0, U, U_i 
    
    def Island_main(self,data0,data1,data2,data3):#line data, line_status, Load_factor, bus data
        #data0,data1,data2,data3 = Line,LineStatus,Load_fact,Bus
        T_isl = len(data2) #time duration in which islanding is examined
        num_bus = len(data3) #number of lines of the network
        new_dict = dict.fromkeys([i for i in range(0,T_isl)]) #Graphs are saved as dictionaries, keys: hours starting from 0 
        Base_graph = nx.Graph([(data0[j,1],data0[j,2]) for j in range(0,len(data0))]) #for j in range(0,len(C))
        #for each hour create the network of nodes and edges (bus and line)
        D0 = dict.fromkeys([i for i in range(0,T_isl)]) # graph informations: number of segments formed
        A_temp = np.zeros((T_isl)) 
        
        for i in range(0,T_isl):      
            C = np.delete(data0,np.where(data1[i,:]<1)[0],axis=0)  #line outages as data1 contains only 0 and 1
            G = nx.Graph([(C[j,1],C[j,2]) for j in range(0,len(C))]) 
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            new_dict[i] = G.subgraph(Gcc[0])
            
            D0[i] = Base_graph.nodes-new_dict[i].nodes  # the lone nodes #new_dict[0]  
        for i in range(0,T_isl):   
            A_temp[i] = len(list(nx.connected_components(new_dict[i])))
            
        A2 = A_temp.astype(int)
        temp = np.max(A_temp).astype(int)
        A1 = np.zeros((T_isl,temp))
        for i in range(0,T_isl):
            A3 = A2[i]
            for j in range(0,A3):
                A1[i,j] = len(list(nx.connected_components(new_dict[i]))[j])
        a_idx = np.argsort(-A1)
        A4 = np.take_along_axis(A1, a_idx, axis=1)
        
        U0,U_ind = np.unique(A4,axis=0,return_index=True)
        U_i = np.flip(U_ind) 
        U = np.flipud(U0)
          
        return new_dict,D0, U, U_i


    
    def Net_info(self,Bus, Line, Gen, LineStatus, Load_fact, new_dict, U, U_i):
        dict_B = {} #dictionary of bus: segment numbers as keys
        dict_L = {} #dictionary of lines: segment numbers as key
        dict_G = {}
        dict_S = {}
        dict_g_flow = {}
        a0,a1 = LineStatus.shape
        LS_index = np.zeros((a0+1,a1)).astype(int)
        LS_index[0,:] = range(1,a1+1)
        LS_index[1:,:] = LineStatus
        HH0 = np.zeros((len(U_i),2)).astype(int)
        HH0[:,0] = U_i.astype(int)
        HH0[:-1,1] = U_i[1:].astype(int)
        HH0[-1,1] = int(len(Load_fact))
        for h in range(0,len(U_i)):
            net_graph = list(nx.connected_components(new_dict[U_i[h]]))
            net_graph.sort(key=len, reverse=True)
            for j in np.where(U[h]>0)[0]:
                Bus_r = np.array(list(net_graph[j])).astype(int)
                Bus_r = np.sort(Bus_r) #1,2,3,...
                Bus0 = Bus[Bus_r-1,:2] #Bus_r-1 : index
                
                tmp = np.where((np.isin(Line[:,2],Bus_r))&(np.isin(Line[:,1],Bus_r)))[0]
                tmp1 = np.sort(tmp)
                Line0 = Line[tmp1]
                
                
                tmp2 = np.where(np.isin(Gen[:,1],Bus_r))[0]
                tmp2 = np.sort(tmp2)
                Gen0 = Gen[tmp2]
                
                dct_gn = {}
                for b in Bus_r[1:]: #buses in Bus_r starting from index 1 
                    dct_gn[b]= np.where((Gen0[:,1]) == b)[0] #index of gens on that bus with number b
                    
                Dct_g = dict((k, v) for k, v in dct_gn.items() if len(v) > 0) #key: number of original buses, value: index of gen
                
                
                LnIsl = LS_index[:,tmp1] #line status in islanded segments
                
                S_tmp = {}
                for t in range(HH0[int(h),0],HH0[int(h),1]): 
                    S_tmp[t] = LnIsl[0,np.where(LnIsl[t+1,:] == 0)[0]]
                S = dict((k, v) for k, v in S_tmp.items() if len(v) > 0) # line number of line outages
                
                temp3 = str(h)+str(',')+str(j)
                dict_B[temp3] = Bus0        
                dict_L[temp3] = Line0
                dict_G[temp3] = Gen0
                dict_g_flow[temp3] = Dct_g  
                dict_S[temp3] = S
                
        Dct_S = dict((k, v) for k, v in dict_S.items() if len(v) > 0)              
                
        return dict_B,dict_L,dict_G,dict_g_flow,Dct_S,HH0

    def Segment_finder(self,HH2,dict_L,M):    
        seg = {} # the resultant segments
        for n, v in dict_L.items():
            seg.setdefault(n, [])
        
        #all indices: the aaa containts t,m,a:segment number by time,key: segment number
        aaa = np.zeros((1000000,4)).astype(int)
        a0 = 0
        #find segments of each (m,t)
        for t in M.keys():
            for a, b in HH2.items():  
                if t in b: #if any(b == t):
                    t_h = a 
                    aaa[a0,2] = a
            for m in M[t]:#for m,t in M.items():
                aaa[a0,0:2] = t,m            
                for key, val in dict_L.items():
                    if str(t_h)+str(',') in key:                    
                        if m+1 in (val[:,0]).astype(int): 
                            aaa[a0,3] =  list(dict_L).index(key)
                a0 += 1
        
        aaa = aaa[:a0,:]
        aac = np.delete(aaa,2,1).astype(int) # if just instances are to be used
        #unique_rows_02 = np.unique(aac[:,(0,2)], axis=0) 
        unique_rows_12 = np.unique(aac[:,(1,2)], axis=0)
        unique_seg = np.unique(aac[:,-1]).astype(int)
        
        M2 = {}       
        #seg: each dictionary key is the segment name and values are the time index and line index
        for i in unique_seg: 
            ind_l = unique_rows_12[np.where(unique_rows_12[:,1]==i)[0],0]
            M2.setdefault(list(dict_L)[i], [])    
            M3 = {}        
            for i1 in ind_l:
                M3.setdefault(i1,[])
                M3[i1]=aac[np.where((aac[:,1]==i1)&(aac[:,2]==i))[0],0]
            M2[list(dict_L)[i]] = M3
        M2 = dict((k, v) for k, v in M2.items() if len(v) > 0) #seg name:{t:m s}
        return M2


    def PTDF(self,Bus, Line, new_dict, U, U_i):
        dict_sh = {}
        for h in range(0,len(U_i)):
            net_graph = list(nx.connected_components(new_dict[U_i[h]]))
            net_graph.sort(key=len, reverse=True)
            for j in np.where(U[h]>0)[0]:
                Bus_r = np.array(list(net_graph[j])).astype(int)
                Bus_r = np.sort(Bus_r)
                Bus0 = Bus[Bus_r-1,:2]
                
                Key2 = np.zeros_like(Bus0).astype(int)
                Key2[:,0] = np.arange(len(Bus0))+1
                Key2[:,1] = Bus0[:,0] #indexing in the basis bus data (original has indexing not arranged from 0 to number of bus) basis is indexed form 0 to 2000
            
                Bus_sh = np.zeros_like(Bus0)
                Bus_sh[:,0] = np.arange(len(Bus0))+1
                Bus_sh[:,1] = Bus0[:,1]
                tmp = np.where((np.isin(Line[:,2],Bus_r))&(np.isin(Line[:,1],Bus_r)))[0]
                tmp1 = np.sort(tmp)
                Line0 = Line[tmp1]
                Line_sh = np.zeros_like(Line0)
                Line_sh[:,0] = np.arange(len(Line0))+1
                for i in range(0,len(Line0)):
                    Line_sh[i,1] = Key2[np.where(Line0[i,1]==Key2[:,1]),0]
                    Line_sh[i,2] = Key2[np.where(Line0[i,2]==Key2[:,1]),0]
                Line_sh[:,3:] = Line0[:,3:]
                   
            
                K1 = range(len(Line_sh))
                N1 = range(len(Bus_sh))
        
                Bbr = np.zeros((len(Line_sh),len(Line_sh)))#[[0 for x in range(len(K))] for y in range(len(K))]
                B = np.zeros((len(Bus_sh),len(Bus_sh)))#[[0 for x in range(len(N))] for y in range(len(N))]
                A = np.zeros((len(Line_sh),len(Bus_sh)))#[[0 for x in range(len(K))] for y in range(len(K))]
                Ared = np.zeros((len(Line_sh),len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(K))]
                Bred = np.zeros((len(Bus_sh)-1,len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(N)-1)]
                shiftfactor = np.zeros((len(Line_sh),len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(K))]
                
                Bbr = np.diag(Line_sh[:,4])
                Line_no = np.array(Line_sh[:,0]-1,dtype = int)
                from_b = np.array(Line_sh[:,1]-1,dtype = int)
                to_b = np.array(Line_sh[:,2]-1,dtype = int)
                A[Line_no,from_b] = 1
                A[Line_no,to_b] = -1
            
                Ared = A[:,1:]
                
                #calculate the B matrix B-inverse    
                for k in K1:
                    B[int(Line_sh[k,1]-1),int(Line_sh[k,2]-1)] += -Bbr[k,k]
                    B[int(Line_sh[k,2]-1),int(Line_sh[k,1]-1)] += -Bbr[k,k]
                for i in N1:
                    B[i,i] = -np.sum(B[i,:])
                
            
                Bred = B[1:,1:]       
              
                #claculate shift factor matrix    
                inBred = np.linalg.inv(Bred)
                shiftfactor = np.matmul(np.matmul(Bbr,Ared),inBred)
                temp2 = str(h)+str(',')+str(j)
                dict_sh[temp2] = shiftfactor
        
        return dict_sh

    

    def vis_img(self,data0):
        
        path = os.getcwd()
        folder_name = self.out_im_f
        path = os.path.join(path, folder_name)
        os.makedirs(path, exist_ok=True)
        # result_folder = f'{os.getcwd()}/{self.out_f}/{data0}'
        #data0 = 'LoadShedding5.xlsx'
        xlsload = pd.ExcelFile(data0)
        df0 = pd.read_excel(xlsload, 'Lshedding_loc',index_col=0).fillna(0)
        H_dur = df0.shape[1] - 4
        for i in range(0,int(H_dur)):
            item = i+1
            fig = plt.figure()
            ax = fig.subplots()
            
            item2 = str('Load Shedding in each zone at hour ')+str(i+1)
            ax.set_title(item2)
                
            # Set the dimension of the figure
            plt.rcParams["figure.figsize"]=8,8;
        
            # Make the background map
            m=Basemap(llcrnrlon=-107, llcrnrlat=25.6, urcrnrlon=-93.5, urcrnrlat=36.8)
            m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
            m.fillcontinents(color='grey', alpha=0.3)
            m.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
            m.drawstates(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
            m.drawcoastlines(linewidth=0.1, color="white")
        
            # prepare a color for each point depending on the continent.
            df0['labels_enc'] = pd.factorize(df0['Zone'])[0]
        
            # Add a point per position
            m.scatter(
                x=df0['Longt'], 
                y=df0['Lat'], 
                s=df0[item]*3,
                alpha=0.75, 
                c=df0['labels_enc'], 
                cmap="Set1")  
        
            mol = str('Hour')+str(i+1)
            fig.savefig(f'{path}/{mol}.png', bbox_inches='tight', dpi=1200)
            plt.close(fig) 
            # fig.savefig(mol+".png")     
        return None

