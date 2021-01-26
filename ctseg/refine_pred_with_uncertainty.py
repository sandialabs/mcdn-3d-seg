from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

def refine_pred(thresh, img, pred, pred_std, flip_bit):
    pred_new = pred.copy()
    if flip_bit:
        pred_new[pred_std[:,:,:,0]>=thresh]=np.bitwise_xor(pred_new[pred_std[:,:,:,0]>=thresh], 1)
    else:
        pred_new[pred_std[:,:,:,0]>=thresh]=0
    return pred_new

def naive_pred(naive_thresh, img):
    pred_new = np.zeros(img.shape)
    pred_new[img[:,:,:]>=naive_thresh]=1
    return pred_new

def score_pred(img, pred):
    mean0 = img[pred==0].mean()
    mean1 = img[pred==1].mean()
    return mean0-mean1

def get_thresh(thresh, img, pred, pred_std, flip_bit):
    pred_new = refine_pred(thresh, img, pred, pred_std, flip_bit)
    score = score_pred(img, pred_new)
    return score

def get_naive_thresh(naive_thresh, img):
    pred_new = naive_pred(naive_thresh, img)
    score = score_pred(img, pred_new)
    return score    

def find_scale(img, pred, pred_std, flip_bit):
    thresh = 0
    best_score = 0
    best_thresh = thresh
    for i in range(15):
        pred_new = refine_pred(thresh, img, pred, pred_std, flip_bit)
        score = score_pred(img, pred_new)
        print(f"Best score: {best_score} Best thresh: {best_thresh}")
        print(f"Score: {score} Thresh: {thresh}")
        if score <= best_score:
            best_score = score
            best_thresh = thresh
        thresh = thresh-2
    return best_thresh
        
def process_pred(img, pred, pred_std, flip_bit, refine_save_path, thresh_save_path):
    pred_std = np.log(pred_std)
    print("Optimizing uncertainty threshold")
    start_time = datetime.now()
    scale = find_scale(img, pred, pred_std, flip_bit)
    scale_time = datetime.now() 
    print(f"Finding best scale took {scale_time - start_time}")
    opt = minimize_scalar(get_thresh,args=(img, pred, pred_std, flip_bit), method='bounded', bounds=(scale-2,min(0, scale+2)), tol=1e-6)
    print(f"Optimizing took {datetime.now() - scale_time}")
    thresh = opt.x
    print(f"Best uncertainty threshold: {thresh}")
    pred_refined = refine_pred(thresh, img, pred, pred_std, flip_bit)
    np.save(refine_save_path, pred_refined)
    print("Optimizing pixel value threshold")
    start_time = datetime.now()
    opt = minimize_scalar(get_naive_thresh,args=(img),method='bounded', bounds=(0,1), tol=1e-3)
    print(f"Optimizing pixel threshold took {datetime.now() - start_time}")
    naive_thresh = opt.x
    print(f"Best naive threshold: {naive_thresh}")
    pred_thresh = naive_pred(naive_thresh, img)
    np.save(thresh_save_path, pred_thresh)
    return pred_refined, pred_thresh
