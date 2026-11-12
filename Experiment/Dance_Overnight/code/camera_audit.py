"""Measure image geometry against the calibration view using stationary background features.

The initial visual suspicion of an ending zoom was rejected by this audit.
"""
from pathlib import Path
import cv2,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';D=OUT/'camera_audit';D.mkdir(exist_ok=True)

def main():
    cv2.setNumThreads(4);cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,4420-3250);ok,reference=cap.read();assert ok;mask=np.zeros(reference.shape[:2],np.uint8);mask[30:320,30:1250]=255;sift=cv2.SIFT_create(nfeatures=1800);kr,dr=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),mask);matcher=cv2.BFMatcher();rows=[];cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for k in range(2424):
        ok,im=cap.read();assert ok
        if k%5 and k!=2423:continue
        kp,dd=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask)
        if dd is None or len(dd)<5:
            rows.append({'source_frame':3250+k,'matches':0,'inliers':0,'current_to_reference_affine':None});continue
        pairs=matcher.knnMatch(dd,dr,k=2);good=[m for m,n in pairs if m.distance<.7*n.distance]
        if len(good)<5:
            rows.append({'source_frame':3250+k,'matches':len(good),'inliers':0,'current_to_reference_affine':None});continue
        a=np.float32([kp[m.queryIdx].pt for m in good]);b=np.float32([kr[m.trainIdx].pt for m in good]);matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000,confidence=.995);row={'source_frame':3250+k,'matches':len(good),'inliers':int(inliers.sum()) if inliers is not None else 0,'current_to_reference_affine':matrix.tolist() if matrix is not None else None}
        if matrix is not None:
            p=np.array([[314.44,395.5],[987,400.43],[100,550],[1180,550],[640,480]],np.float32);warped=p@matrix[:,:2].T+matrix[:,2];row['max_stage_pixel_displacement']=float(np.max(np.linalg.norm(warped-p,axis=1)));row['scale_current_to_reference']=float(np.sqrt(np.linalg.det(matrix[:,:2])));row['background_inlier_rmse']=float(np.sqrt(np.mean(np.sum((a@matrix[:,:2].T+matrix[:,2]-b)[inliers.ravel().astype(bool)]**2,axis=1))))
        rows.append(row)
        if k%250==0:print('Camera audit',3250+k,flush=True)
    cap.release();(D/'results.json').write_text(json.dumps({'reference_source_frame':4420,'scope':'SIFT matches in stationary upper background only, robust partial affine registration. This diagnoses camera geometry; it does not establish physical stage dimensions. No dancer labels or detector predictions used.','rows':rows},indent=2));valid=[r for r in rows if r.get('inliers',0)>=20];fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True);x=[r['source_frame']/25 for r in valid];ax[0].plot(x,[r['scale_current_to_reference'] for r in valid]);ax[0].set_ylabel('Scale to calibration view');ax[1].plot(x,[r['max_stage_pixel_displacement'] for r in valid]);ax[1].set_ylabel('Max anchor shift (pixels)');ax[1].set_xlabel('Original video time (seconds)');fig.suptitle('Stationary-background audit: small camera drift, no ending zoom');fig.tight_layout();fig.savefig(D/'camera_motion.png',dpi=160);plt.close(fig);print('Camera geometry summary',[(r['source_frame'],round(r['scale_current_to_reference'],4),round(r['max_stage_pixel_displacement'],1)) for r in valid if r['source_frame']%100==0],flush=True)

if __name__=='__main__':main()
