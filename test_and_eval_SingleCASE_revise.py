# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:19:46 2018

@author: yang
"""



import pickle
import os
import argparse
import tensorflow as tf
import numpy as np

from model_test import Model
from TestDataloader import TestDataloader
import shutil



def get_absolute_trajectory(traj):
    '''
    Function that computes the absolute coordinates of the trajectory
    traj: (10,7)
    '''
    new_traj=np.zeros([len(traj),1])
    
    for i in range(len(traj)):
         
        real_lat=traj[i,0]*(traj[i,1]-traj[i,2])+traj[i,2]  # lat        
        new_traj[i,0]=real_lat

    return new_traj
    


def get_mean_lat_error(predicted_traj, true_traj, observed_length):
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, 0:1]
        # The true position
        true_pos = true_traj[i, 0:1]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    return np.mean(error)



# ----------------------------------------basic parameters----------------------------#
parser=argparse.ArgumentParser()
parser.add_argument('--obs_length',type=int,default=8, 
                    help='Observed length of the trajectory')
parser.add_argument('--pred_length',type=int,default=8, 
                    help='Predicted length of the trajectory')

parser.add_argument('--seq_length',type=int,default=16, 
                    help='Number of test cases')

parser.add_argument('--seq_interval', type=int, default=1,
                    help='time interval when picking up sequences')

parser.add_argument('--model_checkpoint', type=str, default=None,
                    help='model checkpoint')
                    
parser.add_argument('--saving_path', type=str, default=None,
                    help='saving path')
                    
parser.add_argument('--epoch', type=str, default=None,
                    help='epoch')

parser.add_argument('--test_case', type=int, default=None,
                    help='ID of testccase')
                    
args=parser.parse_args()



if __name__ == '__main__':

    #-------------------------------------- parameters setting----------------------#
    print("args:",args)
    #  original
    model_checkpoint = args.model_checkpoint
    model_checkpoint = model_checkpoint[:-6] # eg. epoch_2_model.ckpt-196.index -> eg. epoch_2_model.ckpt-196
    print("model_checkpoint:", model_checkpoint)

    saving_path =args.saving_path
    epoch = args.epoch
       
    #------------------------------------Loading cache ------------------------------#
    with open(os.path.join(saving_path, 'config.pkl'), 'rb') as fid: 
        saved_args = pickle.load(fid)
     
    test_Dataloader=TestDataloader([args.test_case],args.seq_length, args.seq_interval)
    INPUT_test=test_Dataloader.X
 
    

    #------------------ model setting ----------------#
    tf.reset_default_graph()
    model=Model(saved_args,True)

    
    sess = tf.InteractiveSession() #Initialize TensorFlow session
    saver = tf.train.Saver() # Initialize TensorFlow saver
    saver.restore(sess, model_checkpoint) #restore the model at the checkpoint


    #------------------Testing ------------------------#
    num_batches=len(INPUT_test)
    print('num_batches of the test case:',num_batches)
    
    # test errors         

    lat_error_inv=0.0
    lat_error_inv_fp=0.0

    error_each_inv_lat=0.0
    error_each_inv_lat_fp=0.0

    all_trajectory=[]
    lat_error_box=[]
    TTE_0_flag=1000
    
    if num_batches==0:
        print('num_batches==0')
        
   #-----------------------one by one------------------#  
    keybox=[]
    for b in range(1, num_batches+1): 
        
        batch_x=np.array(INPUT_test[b-1:b],dtype=np.float32)
        batch_x=batch_x[0]      
        obs_traj = batch_x[:args.obs_length] # observed trajectory
        pred_traj_tte_fp=batch_x[-1,3]
        print('TTE of the final point:',pred_traj_tte_fp)
        if pred_traj_tte_fp==0.0:
            TTE_0_flag=b
        
        pre_traj_tte=batch_x[args.obs_length:args.seq_length,3]
        print('TTE of the whole predicted trajectory:',pre_traj_tte)
        
        if 0 in pre_traj_tte:
            print('this includes zero!')
            keybox.append(b)
                 
        complete_traj = model.sample(sess, obs_traj, num=args.pred_length) 
        
        #-----------------error -----------------------------------------#
                
        #-----------------with inverse-normalization---------------#
        gt_traj_inv=get_absolute_trajectory(batch_x)
        complete_traj_inv=get_absolute_trajectory(complete_traj)
        #------------------errors----------------------------------#
        
        lat_error_inv+=get_mean_lat_error(gt_traj_inv, complete_traj_inv, args.obs_length)        
        lat_error_inv_fp+=get_mean_lat_error(gt_traj_inv, complete_traj_inv, args.seq_length-1)
    
        #---------------save the trajectory-----------------------------#        
       
        lat_error_box.append([get_mean_lat_error(gt_traj_inv, complete_traj_inv, args.obs_length)])        
        all_trajectory.append([batch_x,complete_traj,gt_traj_inv,complete_traj_inv])
    
    error_each_inv_lat=lat_error_inv/num_batches # lat
    error_each_inv_lat_fp=lat_error_inv_fp/num_batches #  lat+fp


    result_str= epoch+","+str(error_each_inv_lat)+","+str(error_each_inv_lat_fp)+'\n'
    print('result_str',result_str)
    
    
    # this deals with final point with TTE=0
    case_save_path=saving_path+'/'+str(args.test_case)
    if os.path.exists(case_save_path):
        shutil.rmtree(case_save_path, ignore_errors=True) 
    
    if not os.path.exists(case_save_path):
        os.makedirs(case_save_path)
    
    if TTE_0_flag!=1000:

        example_traj=all_trajectory[TTE_0_flag-1]  
            
        save_data = os.path.join(case_save_path,'predicted_final_tte.pkl')
        with open(save_data,'wb') as fid:
            pickle.dump(example_traj, fid)
 
    else:
        print('Cannot  find TTE=0 at final point! ')
    
    # this deals with predicted trajectory that includes TTE=0
    
    print('key_box',keybox)
    
    keybox=np.array(keybox)
    
    case_save_path_sub=saving_path+'/'+str(args.test_case)+'/'+'TTE0'
    
    if os.path.exists(case_save_path_sub):
        shutil.rmtree(case_save_path_sub, ignore_errors=True) 
    
    if not os.path.exists(case_save_path_sub):
        os.makedirs(case_save_path_sub)
    

    if len(keybox):

        for i in range(len(keybox)):
            
            temp_traj=all_trajectory[keybox[i]-1]
                     
            save_contents=os.path.join(case_save_path_sub,'predicted_tte'+'_'+str(i)+'.pkl')
            
            with open(save_contents,'wb') as fid:
                pickle.dump(temp_traj,fid)
            
    else:
        
        print('Cannot  find TTE=0 for all predicted trajectory')
            
        
    


