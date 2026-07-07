#%% Importing Libreries
from pathlib import Path
import numpy as np
import cv2 as cv
import h5py
import torch
from segment_anything_hq import SamPredictor, sam_model_registry
from controlnet_aux import HEDdetector
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def getROI(param, idx, shape):
    """
    Definisce una ROI dinamica e periodica interpolando i punti.
    """
    indices = sorted(param.keys())
    if not indices:
        return np.zeros(shape[:2], dtype=np.uint8), None
    if idx <= indices[0]:
        target_pts = np.array(param[indices[0]], dtype=np.float32)
    elif idx >= indices[-1]:
        target_pts = np.array(param[indices[-1]], dtype=np.float32)
    else:
        target_pts = None
        for i in range(len(indices) - 1):
            t0 = indices[i]
            t1 = indices[i+1]
            if t0 <= idx <= t1:
                alpha = (idx - t0) / (t1 - t0)
                p0 = np.array(param[t0], dtype=np.float32)
                p1 = np.array(param[t1], dtype=np.float32)
                target_pts = p0 + (p1 - p0) * alpha
                break
    mask = np.zeros(shape[:2], dtype=np.uint8)
    int_pts = target_pts.astype(np.int32)
    cv.fillPoly(mask, [int_pts], 255)
    return mask, int_pts
def getBetterImg(img, conf):
    """
    Migliora il contrasto e la nitidezza dell'immagine leggendo i parametri dal config.
    """
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    denoised = cv.bilateralFilter(
        img, 
        d=conf.Bilateral.d, 
        sigmaColor=conf.Bilateral.sigmaColor, 
        sigmaSpace=conf.Bilateral.sigmaSpace)
    clahe = cv.createCLAHE(
        clipLimit=conf.CLAHE.clipLimit, 
        tileGridSize=conf.CLAHE.tileGridSize)
    contrasted = clahe.apply(denoised)
    blur = cv.GaussianBlur(contrasted,
        conf.UnsharpMask.kernel, conf.UnsharpMask.sigma)
    sharpened = cv.addWeighted(
        contrasted, 
        1.0 + conf.UnsharpMask.strength, 
        blur, 
        -conf.UnsharpMask.strength, 
        0)
    return sharpened
def getEntropy(img, conf):
    entr_raw = entropy(cv.convertScaleAbs(img), disk(conf.Entropy.disk_size))
    entr = np.clip((entr_raw / 8.0) * 255, 0, 255).astype(np.uint8)
    return entr.astype(np.uint8)
def getEdge(img1, img2, edge_model, conf):
    edge = np.zeros_like(img1)
    if edge_model is not None:
        edge_F = np.array(edge_model(
            cv.cvtColor(img1, cv.COLOR_GRAY2RGB), 
            detect_resolution=conf.HED.detect_resolution, scribble=False))
        edge_B = np.array(edge_model(
            cv.cvtColor(img2, cv.COLOR_GRAY2RGB), 
            detect_resolution=conf.HED.detect_resolution, scribble=False))
        edge = cv.resize(
            cv.subtract(
                cv.cvtColor(edge_F, cv.COLOR_RGB2GRAY),
                cv.cvtColor(edge_B, cv.COLOR_RGB2GRAY)),
            (img1.shape[1], img1.shape[0]),
            interpolation=cv.INTER_LINEAR)
    return edge.astype(np.uint8)
def focusAttention(raw_F, raw_B, roi, edge_model, conf):
      betterB = getBetterImg(raw_B, conf=conf.Enhancement)
      betterF = getBetterImg(raw_F, conf=conf.Enhancement)
      entrF, entrB = getEntropy(betterF, conf=conf.Focus), getEntropy(betterB, conf=conf.Focus)
      diff = cv.subtract(betterF, betterB)
      entr = cv.subtract(entrF, entrB)
      edge = getEdge(betterF, betterB, edge_model = edge_model, conf=conf.Focus)
      attention_map = np.clip(
            (entr.astype(np.float32) * conf.Focus.Weights.entropy) + \
            (edge.astype(np.float32) * conf.Focus.Weights.hed) + \
            (diff.astype(np.float32) * conf.Focus.Weights.diff),
            0, 255).astype(np.uint8)
      if roi is not None:
            mask = cv.bitwise_and(attention_map, attention_map, mask=~roi)
            thresh_val, _ = cv.threshold(mask.reshape(-1, 1), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            _, binary = cv.threshold(attention_map, thresh_val, 255, cv.THRESH_BINARY)
            binary = cv.bitwise_and(binary, binary, mask=roi)
      else:
            _, binary = cv.threshold(attention_map, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
      dilateKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, conf.Focus.Morph.Kernels.Dilate)
      erodeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, conf.Focus.Morph.Kernels.Erode)    
      canny = cv.Canny(binary, 50, 150)
      dilateCanny = cv.dilate(canny, dilateKernel, iterations=conf.Focus.Morph.Its.Dilate)
      contours, _ = cv.findContours(dilateCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      final_mask = np.zeros_like(binary)
      min_area = conf.Focus.Morph.AreaMin
      for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area:
                  cv.drawContours(final_mask, [contour], -1, 255, cv.FILLED)
      focus = cv.erode(final_mask, erodeKernel, iterations=conf.Focus.Morph.Its.Erode)
      sanity = cv.absdiff(
            cv.bitwise_and(raw_F, raw_F, mask=focus),
            cv.bitwise_and(raw_B, raw_B, mask=focus))
      nonZero = sanity[sanity > 0]
      if len(nonZero) > 0:
            p = np.percentile(nonZero, 100-conf.Focus.EmptyCheck.TopPercent) 
            topDiff = nonZero[nonZero >= p]
            if np.mean(topDiff) < conf.Focus.EmptyCheck.MeanThresh:
                  focus = np.zeros_like(focus)
      return attention_map, focus
def getSmartPrompt(focusMap, config):
    if not cv.countNonZero(focusMap): return []
    h_img, w_img = focusMap.shape
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed_map = cv.morphologyEx(focusMap, cv.MORPH_CLOSE, kernel)
    dist_global = cv.distanceTransform(focusMap, cv.DIST_L2, 5)
    contours, _ = cv.findContours(closed_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sam_prompts = []
    def subsample(pts_list, max_pts):
        if max_pts <= 0: return []
        if len(pts_list) > max_pts:
            indices = np.linspace(0, len(pts_list) - 1, max_pts, dtype=int)
            return [pts_list[i] for i in indices]
        return pts_list
    for contour in contours:
        area = cv.contourArea(contour)
        if area < config.MinArea: continue
        x, y, w, h = cv.boundingRect(contour)
        x1, y1 = max(0, x - config.NegativeRing), max(0, y - config.NegativeRing)
        x2, y2 = min(w_img - 1, x + w + config.NegativeRing), min(h_img - 1, y + h + config.NegativeRing)
        xs, ys = np.linspace(x1, x2, config.GridSize, dtype=int), np.linspace(y1, y2, config.GridSize, dtype=int)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.column_stack((xv.ravel(), yv.ravel()))
        mask_vals = closed_map[pts[:, 1], pts[:, 0]]
        dist_vals = dist_global[pts[:, 1], pts[:, 0]]
        pos_candidates = pts[(mask_vals > 0) & (dist_vals >= config.SafeMargin)].tolist()
        neg_candidates = pts[(mask_vals == 0)].tolist()
        pos_pts = subsample(pos_candidates, config.MaxPosPts)
        neg_pts = subsample(neg_candidates, config.MaxNegPts)
        if not pos_pts and config.MaxPosPts > 0:
            _, _, _, max_loc = cv.minMaxLoc(dist_global[y:y+h, x:x+w])
            pos_pts = [[max_loc[0] + x, max_loc[1] + y]]
        sam_prompts.append({
            "area": area,
            "bbox": [x, y, x + w, y + h],
            "point_coords": pos_pts + neg_pts,
            "point_labels": [1] * len(pos_pts) + [0] * len(neg_pts)})
    return sorted(sam_prompts, key=lambda i: i["area"], reverse=True)
def segmentEngine(img, predictor, prompts):
    masks = []
    predictor.set_image(cv.cvtColor(img, cv.COLOR_GRAY2BGR))  
    for prompt in prompts:
        input_box = np.array(prompt['bbox'])
        input_points = np.array(prompt['point_coords'])
        input_labels = np.array(prompt['point_labels'])
        mask, scores, _ = predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            box=input_box[None, :],
            multimask_output=True)
        masks.append(mask[np.argmax(scores)])
    return masks
def groupMasks(masks, attentionMap, roi, config):
    """
    1. Taglia maschere in base a ROI.
    2. Raggruppa per IoU usando DFS nativo.
    3. Assegna Score calcolato tramite Intersection over Union (IoU) con l'attentionMap.
    """
    roi_bool, att_bool = roi > 0, attentionMap > 0 
    objects = []
    for mask in masks:
        mask_in_roi = (mask > 0) & roi_bool
        if not mask_in_roi.any(): 
            continue
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_in_roi.astype(np.uint8) * 255)
        objects.extend([labels == i for i in range(1, num_labels) if stats[i, cv.CC_STAT_AREA] >= config.AreaMin])
    n = len(objects)  
    if n == 0: return [], []
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            inter = (objects[i] & objects[j]).sum()
            union = (objects[i] | objects[j]).sum()
            if union > 0 and (inter / union) >= config.Similarity:
                adj_matrix[i, j] = adj_matrix[j, i] = True            
    visited, connected_labels, current_label = np.zeros(n, dtype=bool), np.zeros(n, dtype=int), 0
    for i in range(n):
        if not visited[i]:
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    connected_labels[node] = current_label
                    stack.extend(np.where(adj_matrix[node])[0])
            current_label += 1
    groups = [[] for _ in range(current_label)]
    for idx, (label, mask) in enumerate(zip(connected_labels, objects)):
        intersection = (mask & att_bool).sum()
        union = (mask | att_bool).sum()
        score = intersection / float(union) if union > 0 else 0.0
        groups[label].append({'id': idx, 'score': score})
    for g in groups:
        g.sort(key=lambda x: x['score'], reverse=True)
    return groups, objects
def maskCollapse(groups, expanded_masks, roi, collapse_config):
    """
    Estrae le maschere Rank 1, calcola una soglia dinamica basata sul percentile
    e su una percentuale del valore massimo, e fonde i vincitori.
    """
    bool_roi = roi > 0
    if not groups:
        return [], np.zeros_like(bool_roi)
    rank1 = [g[0] for g in groups if len(g) > 0]
    if not rank1:
        return [], np.zeros_like(bool_roi)
    scores = np.array([info['score'] for info in rank1])
    relative_thresh = np.percentile(scores, collapse_config.PercentileThresh)
    max_score = np.max(scores)
    percentage_thresh = max_score * (collapse_config.MinMaxPercentage / 100.0)
    dynamic_thresh = max(relative_thresh, percentage_thresh)
    union = np.zeros_like(bool_roi)
    for info in rank1:
        if info['score'] >= dynamic_thresh:
            mask_id = info['id']
            obj_mask = expanded_masks[mask_id]
            union |= obj_mask
    return [union], union # qui forse ci rimettero le mani ma per il momento lo teniamo cosi (a maschera unica)
def cloudSegmenter(img, mask, config):
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8), 0
    cv_mask = (mask > 0).astype(np.uint8) * 255
    blurred = cv.GaussianBlur(img, tuple(config.BlurKernel), 0) 
    masked_pixels = blurred[mask > 0].reshape(-1, 1) 
    val, _ = cv.threshold(masked_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    relaxed = val * config.Relaxation 
    _, binary = cv.threshold(blurred, relaxed, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, tuple(config.DilateKernel))
    expanded_mask = cv.dilate(cv_mask, kernel, iterations=config.DilateIter)
    region = cv.bitwise_and(binary, expanded_mask)
    _, labels = cv.connectedComponents(region)
    overlapping_labels = np.unique(labels[mask > 0])
    valid_labels = overlapping_labels[overlapping_labels > 0]
    cloud = np.isin(labels, valid_labels).astype(np.uint8) * 255
    return cloud
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase0.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
        dstRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase3.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.__name__)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        period = ppr / blades 
        bounds = settings.Bounds
        model = sam_model_registry[
            task_conf.Settings.Segmenter.Name](
                checkpoint = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.DependeciesRoot / 
            task_conf.Settings.Segmenter.Model /
            task_conf.Settings.Segmenter.Checkpoint))
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(model)
        edge_model = HEDdetector.from_pretrained("lllyasviel/Annotators")
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = list(f.keys())
                groupsF = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
                    for camera in cameras}
                groupsB = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Background]
                    for camera in cameras} 
                proces = [k for k in groupsF[cameras[0]].keys()
                    if any(lower <= (int(k) % period) <= upper for lower, upper in bounds)]
                for key in tqdm(proces, total=len(proces), desc=colored(f'Cavitation Analysis 🚀', 'magenta'), ncols=100):
                    data = dict()
                    phase = int(key) % period
                    for camera in cameras:
                        try:  
                            raw_F = groupsF[camera][key][:].astype(np.uint8) 
                            raw_B = groupsB[camera][key][:].astype(np.uint8)
                            roi, _ = getROI(
                                getattr(task_conf.Settings.DynamicROI, camera), phase, raw_F.shape)
                            focusMap, focusBinary = focusAttention(raw_F, raw_B, roi, edge_model=edge_model, conf=settings)
                            prompts = getSmartPrompt(focusBinary, task_conf.Settings.SmartPrompt)  
                            if len(prompts) > 0:    
                                masks = segmentEngine(raw_F, predictor, prompts)
                                groups, objects = groupMasks(
                                    masks, focusMap, roi, task_conf.Settings.Group)
                                cavities, collapse = maskCollapse(groups, objects, roi, task_conf.Settings.Collapse)
                                cloud = cloudSegmenter(raw_F, collapse, task_conf.Settings.Cloud)
                            else:
                                cavities = [np.zeros_like(raw_F)]
                                collapse = np.zeros_like(raw_F)
                                cloud = np.zeros_like(raw_F)
                            data[camera] = [cavities, cloud, roi]  
                        except Exception as e:
                            data[camera] = [[np.zeros_like(raw_F)], np.zeros_like(raw_F), np.zeros_like(raw_F)] 
                    dst = (dstRoot / f'{task_conf.MetaData.OutputName}_{key}{task_conf.MetaData.OutputExt}')
                    pickle.dump(data, open(dst, 'wb'))
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return 