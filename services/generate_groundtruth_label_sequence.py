import os
import argparse
import numpy as np
from pdb import set_trace as bp

def indices_with_intersecting_durs(seg_time_boundaries, rttm_bins, threshold):
   
    rttm_bins[:,1] += rttm_bins[:,0]

    intersect_values = np.minimum(seg_time_boundaries[1], rttm_bins[:,1]) - np.maximum(seg_time_boundaries[0], rttm_bins[:,0])
    return intersect_values, intersect_values > threshold

def generate_labels(segmentsfile, labelsfiledir, ground_truth_rttm, threshold):
   
    if not os.path.exists(labelsfiledir):
        os.makedirs(labelsfiledir, 0o777)
    
    
    print("\t\t Threshold for generating label is {}".format(threshold))
    segments = np.genfromtxt(segmentsfile, dtype='str')
    utts = segments[:,0]
    segments = segments[:,1:]
    filenames = np.unique(segments[:,0])
    segment_boundaries =[]
    utts_filewise =[]
    for f in filenames:
        segment_boundaries.append((segments[:,1:][segments[:,0]==f]).astype(float))
        utts_filewise.append((utts[segments[:,0]==f])) 
        
    gt_rttm = np.genfromtxt(ground_truth_rttm, dtype='str')
    rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
    
    for i,f in enumerate(filenames):
        labelsfilepath = os.path.join(labelsfiledir, "labels_{}".format(f))
        if os.path.isfile(labelsfilepath):
            continue
        labels = open(labelsfilepath,'w')
        if i % 5 == 0:
            print("\t\t Generated labels for {} files".format(i))
       

        labels_f = []
        flag = []
         
        rttm = gt_rttm[gt_rttm[:,1] == f] 
        rttm = rttm[:, rttm_idx]
        
        for j in range(len(segment_boundaries[i])):
            _, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float), threshold)
            labels_f = rttm[label_idx][:,2]

            if np.sum(label_idx) > 2:
                label_f = np.unique(labels_f)                

            elif np.sum(label_idx) == 0:
                intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float),0)
                labels_f = rttm[np.argmax(intersect_values)][2]
                labels_f = np.array([labels_f])
                


            towrite= "{} {}\n".format(utts_filewise[i][j], ' '.join(labels_f.tolist()))
            
            labels.writelines(towrite)
  
        
        labels.close()
    print('DONE with labels')


if __name__=="__main__":

 
    default_dataset="callhome1"
    threshold = 0.75
    default_segments = "../lists/{}/tmp/segments".format(default_dataset)
    default_gt_rttm = "data/{}/rttm".format(default_dataset)
    default_labels_dir = "../ALL_CALLHOME_GROUND_LABELS/{}/threshold_{}".format(default_dataset,threshold)

    print("In the label generation script...")
    parser = argparse.ArgumentParser(description='Speaker Label generation for embeddings')
    # Commenting line 83 because there is no provision to send dataset variable in the function generate label
    # parser.add_argument('--dataset', default=default_dataset, type=str, help='dataset', nargs='?')
    parser.add_argument('--segmentsfile', default=default_segments, type=str, metavar='PATH', help='path of the embedding segments file', nargs='?')
    parser.add_argument('--labelsfiledir', default=default_labels_dir, type=str, metavar='PATH', help='path of the labels file', nargs='?')
    parser.add_argument('--ground_truth_rttm', default=default_gt_rttm, type=str, metavar='PATH', help='path of the ground truth rttm file', nargs='?')
    parser.add_argument('--threshold', default=threshold, type=float, metavar='N', help='threshold duration to assign label')

    args = parser.parse_args()
    generate_labels(**vars(args))




    
