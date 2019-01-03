# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:21:26 2018

@author: yang
"""



import pickle
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math


#%%
parser=argparse.ArgumentParser()

parser.add_argument('--obs_length',type=int,default=8, help='Observed length of the trajectory')

parser.add_argument('--pred_length',type=int,default=8, help='Predicted length of the trajectory')

parser.add_argument('--seq_length',type=int,default=16, help='Number of test cases')

parser.add_argument('--test_case', type=int, default=None,help='ID of testccase')
                    
args=parser.parse_args() 

#%%

def get_mean_lat_error(predicted_traj, true_traj, observed_length):
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i]
        # The true position
        true_pos = true_traj[i]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    return np.mean(error)



#%%  this deals with predictde paart with TTE=0

#for each .pkl file

allresult=[]

for i in range(len(os.listdir('save'+str(0)+'/'+str(args.test_case)+'/'+'TTE0'))):
    
    gt_s=np.zeros([args.seq_length,4])
    pre_s=np.zeros([args.seq_length,4])
    
    # for each test in the cross-validation
    
    for k in range(1):
        
        saving_path_sub='save'+str(k)+'/'+str(args.test_case)+'/'+'TTE0'   #save0/57/TTE0
        
        list_me = os.listdir(saving_path_sub) #save0/57/TTE0, list all files

        path = os.path.join(saving_path_sub,list_me[i])
         
        if os.path.isfile(path):
            
            with open(path,'rb') as fid:
                
                data=pickle.load(fid)

            temp_time=data[0][:,3].reshape([-1,1])  
            temp_gt=np.transpose(data[2].reshape(-1))  # (16)
            temp_pred=np.transpose(data[3].reshape(-1)) # predicted trajectory
            
            gt_s[:,k]=temp_gt
            pre_s[:,k]=temp_pred
            
    gt_mean=np.mean(gt_s,axis=1,keepdims=True) #[16,1]
    pred_mean=np.mean(pre_s,axis=1,keepdims=True) 
    allresult.append([temp_time,gt_mean,pred_mean])
    
# plot the figures

for j in range(len(allresult)):
    
    tem_A=np.reshape(np.array(allresult[j]),[-1,16])
    plt.figure(j)
    l11,=plt.plot(tem_A[0,:], tem_A[1,:],color='green', marker='o')
    l21,=plt.plot(tem_A[0,:], tem_A[2,:],marker='x',color='blue')
    l31,=plt.plot(tem_A[0,args.obs_length:args.seq_length],tem_A[2,args.obs_length:args.seq_length],marker='*',color='red')

    plt.legend(handles = [l11, l21, l31], labels = ['GT','PT-All','PT-P'], loc = 'best')
    plt.xlabel('Time to event [frame]')
    plt.ylabel('Lateral Position [m]')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
  
    #plt.show()

    lat_error_inv=get_mean_lat_error(tem_A[1,:], tem_A[2,:], args.obs_length)        
    lat_error_inv_fp=get_mean_lat_error(tem_A[1,:], tem_A[2,:], args.seq_length-1)
    print('lat_error_inv',lat_error_inv)
    print('lat_error_inv_fp',lat_error_inv_fp)
    print('\n')
    

#%% this deals with final point with TTE=0

saving_path_final_tte=os.path.join('save'+str(0)+'/'+str(args.test_case),'predicted_final_tte.pkl')

if os.path.exists(saving_path_final_tte):
    
    gt_f=np.zeros([args.seq_length,4])
    pre_f=np.zeros([args.seq_length,4])

    for k in range(1):
    
        saving_path='save'+str(k)+'/'+str(args.test_case)
        
        saving_path_final_tte=os.path.join(saving_path,'predicted_final_tte.pkl')
        
        with open(saving_path_final_tte, 'rb') as fid:
            
            save_DATA=pickle.load(fid)  # [batch_x,complete_traj,gt_traj_inv,complete_traj_inv]
        
        tempa_time=save_DATA[0][:,3]  # time
        tempc=save_DATA[2]  # gt traj
        tempd=save_DATA[3]  # predicted traj

        gt_f[:,k]=tempc[:,0]
        pre_f[:,k]=tempd[:,0]
        
    gt_mean_keytraj=np.mean(gt_f,axis=1,keepdims=True) 
    pre_mean_keytraj=np.mean(pre_f,axis=1,keepdims=True) 

    print('gt_mean_keytraj',np.shape(gt_mean_keytraj))
    print('pre_mean_keytraj',np.shape(pre_mean_keytraj))
    print('time',np.shape(tempa_time))
    
    plt.figure()
    
    l11,=plt.plot(tempa_time,gt_mean_keytraj,color='green', marker='o')
    l21,=plt.plot(tempa_time,pre_mean_keytraj,marker='x',color='blue')
    l31,=plt.plot(tempa_time[args.obs_length:args.seq_length],pre_mean_keytraj[args.obs_length:args.seq_length],marker='*',color='red')
    
    plt.legend(handles = [l11, l21, l31], labels = ['GT','PT-All','PT-P'], loc = 'best')
    plt.xlabel('Time to event [frame]')
    plt.ylabel('Lateral Position [m]')  
    plt.grid()
    plt.show()
      
        
else:
    
    print('Cannot find .pkl with final TTE=0! ')
        







