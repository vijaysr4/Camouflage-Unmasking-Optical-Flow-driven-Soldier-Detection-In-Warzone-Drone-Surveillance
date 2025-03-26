import os
import struct
import numpy as np
import cv2
import csv
from dense_optical_flow import compute_dense_optical_flow
from pytorch_spynet import compute_spynet_flow

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
BASE_DIR = '/Data/cv_Optical_Flow/MPI-Sintel/training'
CLEAN_BASE = os.path.join(BASE_DIR, 'clean')
FLOW_BASE = os.path.join(BASE_DIR, 'flow')
OCC_BASE = os.path.join(BASE_DIR, 'occlusions')
OUTPUT_DIR = 'output/benchmark_all'
PER_DENSE_CSV = os.path.join(OUTPUT_DIR, 'per_frame_dense.csv')
PER_SPYNET_CSV = os.path.join(OUTPUT_DIR, 'per_frame_spynet.csv')
OVERALL_DENSE_CSV = os.path.join(OUTPUT_DIR, 'overall_dense.csv')
OVERALL_SPYNET_CSV = os.path.join(OUTPUT_DIR, 'overall_spynet.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_KEYS = [
    'epe_all','epe_matched','epe_unmatched',
    'd0_10','d10_60','d60_140',
    's0_10','s10_40','s40_plus'
]

def read_flo(path):
    with open(path,'rb') as f:
        tag = struct.unpack('f', f.read(4))[0]
        if abs(tag-202021.25)>1e-3:
            raise ValueError(f"Bad .flo tag in {path}")
        w,h = struct.unpack('ii', f.read(8))
        return np.fromfile(f, dtype=np.float32).reshape((h,w,2))

def compute_errors(gt, est):
    return np.linalg.norm(gt-est, axis=2)

def evaluate_flow(gt_flow, est_flow, occ_mask):
    errors = compute_errors(gt_flow, est_flow)
    mask_all = ~np.isnan(errors)
    mask_mat = mask_all & occ_mask

    boundary = cv2.morphologyEx(occ_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))
    dist = cv2.distanceTransform((boundary==0).astype(np.uint8), cv2.DIST_L2,5)
    speed = np.linalg.norm(gt_flow, axis=2)

    return {
        'epe_all': np.nanmean(errors[mask_all]),
        'epe_matched': np.nanmean(errors[mask_mat]),
        'epe_unmatched': np.nanmean(errors[mask_all & ~occ_mask]),
        'd0_10': np.nanmean(errors[mask_mat & (dist<10)]),
        'd10_60': np.nanmean(errors[mask_mat & (dist>=10)&(dist<60)]),
        'd60_140': np.nanmean(errors[mask_mat & (dist>=60)&(dist<140)]),
        's0_10': np.nanmean(errors[mask_mat & (speed<10)]),
        's10_40': np.nanmean(errors[mask_mat & (speed>=10)&(speed<40)]),
        's40_plus': np.nanmean(errors[mask_mat & (speed>=40)])
    }

def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene','frame_index']+METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)

def write_overall(path, rows):
    overall = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        overall[key] = np.mean(vals) if vals else np.nan
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric','overall_mean'])
        for k,v in overall.items():
            writer.writerow([k, f"{v:.3f}" if not np.isnan(v) else "nan"])

def main():
    dense_rows, spynet_rows = [], []
    scenes = sorted(d for d in os.listdir(CLEAN_BASE) if os.path.isdir(os.path.join(CLEAN_BASE,d)))

    for scene in scenes:
        scene_clean = os.path.join(CLEAN_BASE, scene)
        scene_flow = os.path.join(FLOW_BASE, scene)
        scene_occ = os.path.join(OCC_BASE, scene)
        if not os.path.isdir(scene_flow): continue

        for flo_fname in sorted(f for f in os.listdir(scene_flow) if f.endswith('.flo')):
            idx = int(flo_fname.split('_')[1].split('.')[0])
            img1 = os.path.join(scene_clean, f'frame_{idx:04d}.png')
            img2 = os.path.join(scene_clean, f'frame_{idx+1:04d}.png')
            occ_mask = cv2.imread(os.path.join(scene_occ, f'frame_{idx:04d}.png'), cv2.IMREAD_GRAYSCALE)==0
            gt = read_flo(os.path.join(scene_flow, flo_fname))
            if not all(os.path.exists(p) for p in (img1,img2)): continue

            # Dense
            dense_flow = compute_dense_optical_flow(img1, img2)
            dense_metrics = evaluate_flow(gt, dense_flow, occ_mask)
            dense_metrics.update({'scene':scene,'frame_index':idx})
            dense_rows.append(dense_metrics)

            # SPyNet
            spynet_flow, _ = compute_spynet_flow(img1, img2, OUTPUT_DIR)
            spynet_metrics = evaluate_flow(gt, spynet_flow, occ_mask)
            spynet_metrics.update({'scene':scene,'frame_index':idx})
            spynet_rows.append(spynet_metrics)

    write_csv(PER_DENSE_CSV, dense_rows)
    write_csv(PER_SPYNET_CSV, spynet_rows)
    write_overall(OVERALL_DENSE_CSV, dense_rows)
    write_overall(OVERALL_SPYNET_CSV, spynet_rows)

    print("Benchmark complete — CSVs saved in", OUTPUT_DIR)

if __name__=='__main__':
    main()
