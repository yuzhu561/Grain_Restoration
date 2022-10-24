#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 21:29:44 2022

@author: yuzhu
"""

import numpy as np

from skimage import measure, morphology, exposure, transform, filters, io
from skimage.feature import greycomatrix, greycoprops, peak_local_max
from skimage.segmentation import chan_vese, watershed
from skimage.morphology import convex_hull_image, convex_hull_object, skeletonize, disk, thin
from scipy import ndimage, signal
from scipy.spatial import ConvexHull
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import sys
from pylab import *
import cv2
from skimage.draw import polygon, line

def open_image(img_path):
    if os.path.isdir(img_path):
        img_files=os.listdir(img_path)
        img_files.sort()
        data=[]
        for files in img_files:
            img=io.imread(img_path+'/'+files)         
            data.append(img)
        data=np.array(data)
        print('the shape of this image is, ', data.shape)
    else:
        if os.path.splitext(img_path)[-1]=='.nc':
            nc=Dataset(img_path, 'r')
            try:
                data=nc.variables['segmented']
            except:
                data=nc.variables['tomo']
            data=np.array(data)
        else:
            data=io.imread(img_path)
            #data=np.array(data)
        print('the shape of this image is, ', data.shape)
        
    return data

def Remove_small_objects(img, target_phase, number_of_voxels_threshod, connectivity):   
    #img是一个2维或是3维的array，代表一个分割的图像
    #target_phase is an integer which defines what phase will be processed, if target=None means all phases will be processed
    #number_of_voxels_threshod is a area/volum threshod where less than which the object will be removed, e.g., 1000
    #connectivity是一个连通性参数，取值为1或是2，connectivity=1对角位置不连通，connectivity=2表示对角位置联通
    #该函数返回一个与输入图像同样大小的，小尺寸目标（如小孔隙和颗粒）被取消的分割图像
    img=np.array(img)
    ele=np.unique(img)
    img_shape=img.shape
    if len(list(ele))==2: #if the image is a binary image
        if target_phase!=None:
            if target_phase not in list(ele):
                print('The selected phase is not in the input image, the valid phase are: ', ele)
            else:
                img1=img==target_phase
                img1=morphology.remove_small_objects(img1, number_of_voxels_threshod, connectivity=connectivity)
                img1=img1.astype('u1')
                idx0=np.flatnonzero(img1==0)                 
                idx1=np.flatnonzero(img1==1) 
                backgroud_phase=list(set(ele)-set([target_phase]))[0] 
                img.flat[idx0]=backgroud_phase
                img.flat[idx1]=target_phase
        else:
            img1=img==ele[0]
            img1=morphology.remove_small_objects(img1, number_of_voxels_threshod, connectivity=connectivity)            
            img1=img1.astype('u1')
            img1=img1==0
            img1=morphology.remove_small_objects(img1, number_of_voxels_threshod, connectivity=connectivity)
            img1=img1.astype('u1')
            idx0=np.flatnonzero(img1==0)
            idx1=np.flatnonzero(img1==1)
            img.flat[idx0]=ele[0] 
            img.flat[idx1]=ele[1]                                       

    if len(list(ele))>2: #if the image is a multiphase image 
        if target_phase!=None:
            if target_phase not in list(ele):
                print('The selected phase is not in the input image, the valid phase are: ', ele)
            else:
                imgt=img==target_phase           
                imgt_init=imgt.copy()
                imgt_init=imgt_init.astype('u1')
                #注意remove_small_objects函数要求输入图像是布尔型数据(bool)
                imgt=morphology.remove_small_objects(imgt, number_of_voxels_threshod, connectivity=connectivity) 
                imgt=imgt.astype('u1')
                diff=imgt_init-imgt
                ids=ndimage.distance_transform_edt(diff, return_distances=False, return_indices=True)
                index=np.ravel_multi_index(ids, img_shape)
                index_change=np.flatnonzero(diff==1)
                img.flat[index_change]=img.flat[index]
        else:
            for i in list(ele):
                imgt=img==i
                imgt_init=imgt.copy()
                imgt_init=imgt_init.astype('u1')
                imgt=morphology.remove_small_objects(imgt, number_of_voxels_threshod, connectivity=connectivity) 
                imgt=imgt.astype('u1')
                diff=imgt_init-imgt
                ids=ndimage.distance_transform_edt(diff, return_distances=False, return_indices=True)
                index=np.ravel_multi_index(ids, img_shape)
                index_change=np.flatnonzero(diff==1)
                img.flat[index_change]=img.flat[index]
    return img

def Grain_Partition_Erosion_Dilation(seg, target_phase, remove_small_object_size):

    #seg is a segmented image 2D or 3D
    #initial_seg是一个与seg同样大小的图像，代表原始分割图像，考虑到部分情况下因为骨架溶蚀，局部图像会产生缺失，所以此处用seg找颗粒核，用initial_seg做分割
    #target_phase is a integer denotes the target phase e.g., 0 or 1
    #remove_small_object_size 是一个整数（例如500）或是None， 若是整数则代表小于该数值大小的颗粒会被消除，若是None则不进行消除小噪声颗粒操作
    #输出的是一个与输入图像尺寸相同的矩阵，代表颗粒分割的结果

    img=np.zeros(seg.shape, dtype='u1')
    idx=np.flatnonzero(seg==target_phase)
    img.flat[idx]=1
    distance=ndimage.distance_transform_edt(img)
    #======================erosion==================================================================
    if len(list(img.shape))==3: #if img is 3D
        nx, ny, nz=seg.shape
        structure=np.ones((3,3,3), dtype='u1')
    if len(list(img.shape))==2: #if img is 2D
        structure=np.ones((3,3), dtype='u1')

    img1=img.copy()
    label_1, num_1=ndimage.label(img1, structure)
    location_1=[]
    for i in range(num_1):
        idx_i=np.flatnonzero(label_1==i+1)
        location_1.append(idx_i)
    its=0
    while (its<300):
        img1=morphology.binary_erosion(img1, structure)
        if sum(img1)>0:
            label, num=ndimage.label(img1, structure)

            for p in location_1:
                if sum(label.flat[p])==0:
                    img1.flat[p]=1
            img1=img1.astype('u1')
            label_1, num_1=ndimage.label(img1, structure)

            location_1=[]
            for i in range(num_1):
                idx_i=np.flatnonzero(label_1==i+1)
                location_1.append(idx_i)
        else:
            break

        its=its+1

    img_after_erosion=label_1>0
    img_after_erosion=img_after_erosion.astype('i8')

    #===============extract the centre of residual grains cores=============================
    mass_center=ndimage.measurements.center_of_mass(img_after_erosion, label_1, list(range(1, num_1+1)))
    mass_center=np.array(mass_center)
    mass_center=mass_center.astype('i8')
    print(len(mass_center)) 
    markers=np.zeros(distance.shape, dtype='i8')
    label_seq=np.array(list(range(len(mass_center))))+1
    np.random.shuffle(label_seq)
    m=0
    if len(list(img.shape))==3: 
        for i in range(mass_center.shape[0]):
            markers[mass_center[i][0]][mass_center[i][1]][mass_center[i][2]]=label_seq[m]
            m=m+1 
            
        labels = watershed(-distance, markers, connectivity=np.ones((3, 3, 3)), mask=img)
    if len(list(img.shape))==2: 
        for i in range(mass_center.shape[0]):
            markers[mass_center[i][0]][mass_center[i][1]]=label_seq[m]
            m=m+1

        labels = watershed(-distance, markers, connectivity=np.ones((3, 3)), mask=img)

    #===============Remove small objects=================
    if remove_small_object_size!=None:
        markers1=markers.copy()
        markers1=np.array(markers1>0).astype('u1')
        uni=list(np.unique(labels))
        uni=uni[1:len(uni)]
        for i in uni:
            idx=np.flatnonzero(labels==i)
            if len(idx)<remove_small_object_size:
                tt=np.zeros(markers1.shape, dtype='u1')
                tt.flat[idx]=1
                tt1=morphology.binary_dilation(tt, structure)
                idx1=np.nonzero(np.logical_and(tt1==1, tt==0))
                idx1=np.ravel_multi_index(idx1, markers1.shape)
                neighbours=np.unique(labels.flat[idx1])
                if len(neighbours)==1:
                    labels.flat[idx]=neighbours[0]
                else:
                    labels.flat[idx]=neighbours[1]

    return labels
    
def remove_noise_particles(labels, concaves, contour_points, dominant_points):
    #该函数用于去除一些由于颗粒边界粗糙而引起的过度分割现象，例如如果一个颗粒被其它一个或多个颗粒包围而不与孔隙接触，则认为该颗粒是噪声颗粒
    #或是一个颗粒与孔隙接触的边界上不存在任何凹点，则认为该颗粒是噪声颗粒
    #labels是颗粒已经初步分割好的图像
    #concaves, contour_points, dominant_points分别是凹点，所有的边界点，关键点（凹点和凸点）
    ele=list(np.unique(labels))
    ele=ele[1:]
    rows, cols=labels.shape
    index=np.arange(rows*cols)
    for elec in ele:
        img=np.zeros(labels.shape, dtype='u1')
        idx=np.flatnonzero(labels==elec)
        img.flat[idx]=1
        #--------------------计算颗粒外边界（边界不在颗粒本身上，而是包裹着颗粒）----------------
        imgt=morphology.binary_dilation(img).astype('u1')
        boundary=imgt-img
        idx_boundary=np.flatnonzero(boundary==1)
        boundary_ele=np.unique(labels.flat[idx_boundary]) #颗粒外边界包含的元素
        if 0 not in list(boundary_ele): #对于没有与孔隙接触的颗粒，将其去掉
            edt, inds = ndimage.distance_transform_edt(img, return_indices=True)
            indsx=inds[0,:,:].reshape(rows*cols,)
            indsy=inds[1,:,:].reshape(rows*cols,)
            idx_c=np.ravel_multi_index((indsx, indsy), img.shape)
            labels.flat[index]=labels.flat[idx_c]
          
        concaves_c, contour_points_c, dominant_points_c=concave_points(img, 2.5)
        intersect1=list(set(list(contour_points_c)).intersection(set(list(contour_points))))
        intersect2=list(set(intersect1).intersection(set(list(concaves_c))))
        if len(intersect2)==0:
            imgtt=np.ones(labels.shape, dtype='u1')
            for boundary_ele_c in boundary_ele:
                if boundary_ele_c>0:
                    idx_t=np.flatnonzero(labels==boundary_ele_c)
                    imgtt.flat[idx_t]=0
            edt, inds = ndimage.distance_transform_edt(imgtt, return_indices=True)
            indsx=inds[0,:,:]
            indsy=inds[1,:,:]          
            for p in list(idx):
                labels.flat[p]=labels[indsx.flat[p]][indsy.flat[p]]    
                    
        #--------------------计算颗粒内边界（边界在颗粒本身上）----------------
        imgt=morphology.binary_erosion(img).astype('u1')
        boundary=img-imgt
        idx_boundary=np.flatnonzero(boundary==1)
        concaves_c, contour_points_c, dominant_points_c=concave_points(img, 2.5)
        boundary_seq=[]
        boundary_seq_c=[]
        for p in list(contour_points_c):
            if p in list(idx_boundary):
                boundary_seq_c.append(p)
            else:
                boundary_seq.append(boundary_seq_c)
                boundary_seq_c=[]                
        flag=0
        for seq_c in boundary_seq:
                if len(seq_c)>5:
                    intersect=list(set(seq_c[2:-2]).intersection(set(list(concaves))))
                    if len(intersect)>0:
                        flag=1                        
        if flag==0: #对于没有与孔隙接触的颗粒，将其去掉
            edt, inds = ndimage.distance_transform_edt(img, return_indices=True)
            indsx=inds[0,:,:].reshape(rows*cols,)
            indsy=inds[1,:,:].reshape(rows*cols,)
            idx_c=np.ravel_multi_index((indsx, indsy), img.shape)
            labels.flat[index]=labels.flat[idx_c]  
        
    return labels    

def smooth_boundary(img, epsilon):
    #该方法使用多边形拟合的方法使得边界变得光滑
    #img is a 2d array that present a segmented image with 2 or more phases
    #epsilon是一个控制光滑程度的参数，可以使小数或是整数，值越大越光滑，例如取值5
    #output is a 2d image with the same size of the inputted image
    
    elements=list(np.unique(img)) #obtain the phase number of the image
    img1=elements[-1]*np.ones(img.shape, dtype='u1')
    for ele in elements[:-1]:
        imgt=img==ele
        imgt=imgt.astype('u1')
        contours, hierarchy = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgtt=np.zeros(img.shape, dtype='u1')
        for cnt in contours:
            approx=cv2.approxPolyDP(cnt, epsilon, True)
            r=approx[:,0,1]
            c=approx[:,0,0]
            rr, cc = polygon(r, c)
            imgtt[rr, cc] = 1         
        idx=np.flatnonzero(imgtt==1)
        img1.flat[idx]=ele
        
    return img1

def smooth_boundary1(img, sampling_step):
    #该函数使用抽稀边界点的方法使边界光滑
    #img is a 2d array that present a segmented image with 2 or more phases
    #sampling_step是一个控制光滑程度的整数，表示在边界contours上每隔几个点取一个点，值越大越光滑，例如取值4
    #output is a 2d image with the same size of the inputted image
    
    elements=list(np.unique(img)) #obtain the phase number of the image
    img1=elements[-1]*np.ones(img.shape, dtype='u1')
    #img1=elements[0]*np.ones(img.shape, dtype='u1')
    for ele in elements[:-1]:
        imgt=img==ele
        imgt=imgt.astype('u1')
        contours, hierarchy = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgtt=np.zeros(img.shape, dtype='u1')
        for cnt in contours:
            #approx=cv2.approxPolyDP(cnt, epsilon, True)
            r=cnt[::sampling_step,0,1]
            c=cnt[::sampling_step,0,0]
            rr, cc = polygon(r, c)
            imgtt[rr, cc] = 1         
        idx=np.flatnonzero(imgtt==1)
        img1.flat[idx]=ele        
    return img1

def concave_points(img, epsilon):
    #该函数用于找出一张二值图像上的轮廓线，关键点（包括凹点和凸点）以及凹点
    #img是一个二维数组，代表一张二值图像（1是目标相）
    #epsilon是一个正数，代表对边界的光滑程度，越大越光滑，一般去2，2.5，3等
    #函数返回三个列表，分别代表凹点，轮廓线，关键点（包括凹点和凸点）
    
    contour_points=[] #用来记录所有的contour上的点    
    dominant_points=[] #用来记录所有的关键点，主要是用多边形拟合时的乖点，包括凸点和凹点
    concaves=[] #用来记录凹点
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#获取所有的边界点

    for cnt in contours:
        
        contour_points_c=np.ravel_multi_index(np.transpose(np.fliplr(cnt[:,0,:])), img.shape)
        contour_points_c=list(contour_points_c)
        contour_points=contour_points+contour_points_c

        approx=cv2.approxPolyDP(cnt, epsilon, True)
        dominant_points_c=np.ravel_multi_index(np.transpose(np.fliplr(approx[:,0,:])), img.shape)
        dominant_points_c=list(dominant_points_c)
        dominant_points=dominant_points+dominant_points_c

        if len(dominant_points_c)>2: #对于节点数大于2的线段进行分析
            for i in range(len(dominant_points_c)):
                imgtt=np.zeros(img.shape, dtype='u1')
                mid=dominant_points_c[i]
                if i>0 and i<len(dominant_points_c)-1:
                    left=dominant_points_c[i-1]
                    right=dominant_points_c[i+1]                
                if i==0:
                    left=dominant_points_c[-1]
                    right=dominant_points_c[i+1]
                if i==len(dominant_points_c)-1:
                    left=dominant_points_c[i-1]
                    right=dominant_points_c[0] 
                points=[left, mid, right]
                coordi=np.unravel_index(np.array(points), img.shape)
                 
                r=coordi[0]
                c=coordi[1]
                rr, cc = polygon(r, c)
                imgtt[rr, cc] = 1
        
                idx=np.flatnonzero(imgtt==1)
                difference_idx=list(set(list(idx)).difference(set(contour_points_c)))
                num=len(list(difference_idx))
                if num>0:
                    if np.sum(img.flat[np.array(difference_idx)])<0.5*num:
                        #concaves.append(mid)                
                        leftx=coordi[0][0]
                        lefty=coordi[1][0]
                        midx=coordi[0][1]
                        midy=coordi[1][1]
                        rightx=coordi[0][2]
                        righty=coordi[1][2] 
                        v1=np.array([leftx-midx, lefty-midy]) 
                        v2=np.array([rightx-midx, righty-midy])
                        cos_seta=float(v1[0]*v2[0]+v1[1]*v2[1])/float((np.sqrt(v1[0]**2+v1[1]**2))*(np.sqrt(v2[0]**2+v2[1]**2)))   
                        if cos_seta>(-1)*np.cos(np.pi/7):
                            concaves.append(mid)                        
                
    return concaves, contour_points, dominant_points

def contour_segments(img, concaves):
    #img是一个二值图像（含0，1两相）代表目标图像
    #concaves是一个列表，代表所有的凹点的集合
    #输出的是一个列表，每个元素代表一个以凹点分割的颗粒边界片段

    segments_all=[]
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imgtest=np.zeros(img.shape, dtype='u1')
    for cnt in contours:
        contour_points_c=np.ravel_multi_index(np.transpose(np.fliplr(cnt[:,0,:])), img.shape)
        segments=[]
        segments_c=[] 
        
        inter=list(set(list(contour_points_c)).intersection(set(concaves)))
        if len(inter)==0:
            segments_all.append(list(contour_points_c))
        else:
            for p in list(contour_points_c):
                segments_c.append(p)
                if p in list(concaves):
                    segments.append(segments_c)
                    segments_c=[]
                    segments_c.append(p)
            segments.append(segments_c)
            start=segments[0]
            end=segments[-1]
            segments[-1]=end+start
            del segments[0]

        for p in segments:
            segments_all.append(p)
                         
    return segments_all   


def boundary_segment(img, contour_points):

    #该函数用于找出当前分割结果下某一个颗粒的所有contour segments
    #labels是颗粒已经初步分割好的图像
    #concaves, contour_points, dominant_points分别是凹点，所有的边界点，关键点（凹点和凸点）
    #返回值是一个列表，每个元素代表该颗粒的一段连续边界片段
           
    contour_points_c, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imgtest=np.zeros(img.shape, dtype='u1')
    contour_points_obj=[]
    for cnt in contour_points_c:  
        contour_c_obj=np.ravel_multi_index(np.transpose(np.fliplr(cnt[:,0,:])), img.shape)
        contour_points_obj=contour_points_obj+list(contour_c_obj) 
        imgtest.flat[contour_c_obj]=1
        
    contour_points_c=[]
    for p in contour_points_obj:
        if p not in contour_points_c:
            contour_points_c.append(p)
            
    for p in contour_points_c:
        if p not in contour_points:
            imgtest.flat[p]=0
  
    label, num=ndimage.label(imgtest, np.ones((3,3), dtype='u1'))
    ele=np.unique(label)
    ele=list(ele[1:])
    boundary_seq=[]
    for i in ele:
        idx=list(np.flatnonzero(label==i))
        seq=[]
        for t in idx:
            seq.append(contour_points.index(t))
        seq.sort()
        seq_idx=[]
        for t in seq:
            seq_idx.append(contour_points[t])
   
        boundary_seq.append(list(seq_idx))
    
    return boundary_seq


def contour_estimation(labels, segments_all):
    #该函数用于拟合颗粒边界
    #labels是一个颗粒分割完毕的图像，最好是经过adjust_partitioning处理过的分割图像，因为计算压实作用的影响更精确
    #segments_all是一个列表，其中的每个元素代表一段孔隙-基质边界片段，片段之间以凹点分割
    #输出的图像表现的是恢复颗粒重叠前的形状
    imgs=labels
    rows, cols=imgs.shape
    ele=list(np.unique(imgs))
    ele=ele[1:]
     
    segments_all_pixels=[]
    for p in segments_all:
        segments_all_pixels=segments_all_pixels+p
        
    img_contours=np.zeros(labels.shape, dtype='u1')#*******************************************************    
    selected_properties=['major_axis_length', 'minor_axis_length', 'eccentricity'] #选择将要统计的颗粒几何形态参数
    grain_properties=np.zeros((len(ele), len(selected_properties)+3), dtype='f8')
    i=0   
    for elec in ele:
        imgt=np.zeros(imgs.shape, dtype='u1')
        idx=np.flatnonzero(imgs==elec)
        imgt.flat[idx]=1
        boundary_segments=[]
        seg_seq=[]
        contours, hierarchy = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_points=[]
        for cnt in contours:  
            contour_c=np.ravel_multi_index(np.transpose(np.fliplr(cnt[:,0,:])), imgs.shape)
            contour_points=contour_points+list(contour_c)

        #-----------------------------??????????????????????????????????????????????--------------------------------
        current_boundary_points=[]
        for p in segments_all:
            inter=list(set(p).intersection(set(contour_points)))#计算出该轮廓片段与当前颗粒边界轮廓之前的交集
            similarity=np.float(len(inter))/np.float(len(p)) #计算相似度
            if similarity>0.5 and len(inter)>5:
                current_boundary_points=current_boundary_points+p[3:-3]
                
        for p in contour_points: #将轮廓点按照特定顺序排列
            if p in current_boundary_points:
                boundary_segments.append(p)

        if len(boundary_segments)>5: 
        
            boundary_t=[]
            for t in boundary_segments:
                if t not in boundary_t:
                    boundary_t.append(t)
            boundary_segments=boundary_t
            
            print('the length of boundary vector is: ', len(boundary_t))
            
            #imgt.flat[np.array(boundary_segments)]=20
            x, y=np.unravel_index(np.array(boundary_segments[::2]), imgs.shape)
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
            tck, u = scipy.interpolate.splprep([x, y], s=0, per=True)
            #tck, u = scipy.interpolate.splprep([x, y], k=3, s=10, per=True)
            xi, yi = scipy.interpolate.splev(np.linspace(0, 1, 5000), tck)
            xi=np.round(xi).astype('i4')
            yi=np.round(yi).astype('i4')
            idx_out=list(np.flatnonzero(np.logical_and(xi>0, xi<rows)))
            idy_out=list(np.flatnonzero(np.logical_and(yi>0, yi<cols)))
            idxy_out=np.array(list(set(idx_out).intersection(set(idy_out))))                      
            img_contours[xi[idxy_out], yi[idxy_out]]=1 
            
            #--------------------------计算颗粒几何形态---------------------------
            imgtest=np.zeros(imgt.shape, dtype='u1')
            rr, cc = polygon(xi[idxy_out], yi[idxy_out]) 
            imgtest[rr, cc]=1
            grain_size_compressed=np.sum(imgt) #计算被压实之后的颗粒大小
            grain_size_initial=np.sum(imgtest) #计算被压实之前的颗粒大小
            compress_ratio=float((grain_size_initial-grain_size_compressed))/float(grain_size_initial) #计算压实率            
            props=measure.regionprops_table(label_image=imgtest, properties=selected_properties)
            
            grain_properties[i][0]=grain_size_compressed
            grain_properties[i][1]=grain_size_initial
            grain_properties[i][2]=compress_ratio
            t=0
            for t in range(len(selected_properties)):
                current_prop=props[selected_properties[t]]
                grain_properties[i][t+3]=current_prop[0]
                t=t+1
            #--------------------------------------------------------------------                 
        i=i+1           
     
    return img_contours, grain_properties
    
def adjust_partitioning(labels, segments_all):
    #该函数用于微调分割点的位置使其严格处于凹点位置
    #labels是一个颗粒分割好的图像
    #segments_all所有的孔隙-基质边界轮廓片段（contour_segments），所有的轮廓以凹点分开形成一个个片段
    #输出是一个修整之后的labels图像
    
    imgs=labels.copy()
    rows, cols=imgs.shape
    ele=list(np.unique(imgs))
    ele=ele[1:]
     
    segments_all_pixels=[]
    for p in segments_all:
        segments_all_pixels=segments_all_pixels+p
        
    label_adjusted=np.zeros(labels.shape, dtype='u1')#*******************************************************
    imgtrans=np.zeros(labels.shape, dtype='u1') 
    for elec in ele: #对每一个分割出来的颗粒进行分析
        imgt=np.zeros(imgs.shape, dtype='u1')
        idx=np.flatnonzero(imgs==elec)
        imgt.flat[idx]=1
        boundary_segments=[] #用于存放该颗粒的（凹点-凹点）边界轮廓，即与segments_all匹配的边界轮廓
        seg_seq=[]
        contours, hierarchy = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #找出当前颗粒的轮廓
        contour_points=[] #用于存储当前颗粒所有的轮廓点
        for cnt in contours: #将所有轮廓点顺序存入一个列表（contour_points）中
            contour_c=np.ravel_multi_index(np.transpose(np.fliplr(cnt[:,0,:])), imgs.shape)
            contour_points=contour_points+list(contour_c)

        for p in segments_all: #对每一个边界轮廓片段进行分析
            inter=list(set(p).intersection(set(contour_points))) #计算出该轮廓片段与当前颗粒边界轮廓之前的交集
            #====================注意此时inter中的点不是按照边界轮廓顺序变化的================================
            inter1=[]
            for t in contour_points:
                if t in inter:
                    inter1.append(t)
            inter=inter1
            #===================================inter内部点的顺序调整完毕====================================
            similarity=np.float(len(inter))/np.float(len(p)) #计算相似度
            if similarity>0.5 and len(p)>5:
                boundary_segments.append(p)
                #label_adjusted.flat[p]=elec
                imgtrans.flat[p]=1
                #====判断轮廓片段p与当前颗粒片段的记录顺序（顺时针或是逆时针）是否一致，如不一致则调整为一致=====
                present_p=p[int(np.floor(float(len(p)))/2)] #找出曲线段p的中点的index
                present_p1=p[int(np.floor(float(len(p)))/2)+1]                               
                if len(inter)>1:
                    first=p.index(inter[0])
                    second=p.index(inter[1])
                    if first<second:
                        present_p=p[first]
                        present_p1=p[second]
                    else:
                        present_p=p[second]
                        present_p1=p[first]  
                                                     
                if contour_points.index(present_p)>contour_points.index(present_p1):
                    print('complete a reverse')
                    p.reverse()
                #===============================片段p轮廓点顺序调整一致完毕======================================                
                seg_seq.append(contour_points.index(present_p)) #在当前颗粒边界片段上取一个点，防止该颗粒的边界片段出现顺序不一致
        seg_seq_sort=seg_seq.copy()
        seg_seq_sort.sort()
        boundary_segments_1=[]      
        for p in seg_seq_sort:
            p_index=seg_seq.index(p)
            boundary_segments_1.append(boundary_segments[p_index])            

        for i in range(len(boundary_segments_1)):        
            if i<len(boundary_segments_1)-1:
                end=boundary_segments_1[i][-1]
                start=boundary_segments_1[i+1][0]
                endx=end//cols
                endy=end-endx*cols
                startx=start//cols
                starty=start-startx*cols
                rr, cc=line(endx, endy, startx, starty)
                #label_adjusted[rr, cc]=elec
                imgtrans[rr, cc]=1
            else:
                end=boundary_segments_1[i][-1]
                start=boundary_segments_1[0][0]
                endx=end//cols
                endy=end-endx*cols
                startx=start//cols
                starty=start-startx*cols
                rr, cc=line(endx, endy, startx, starty)
                #label_adjusted[rr, cc]=elec
                imgtrans[rr, cc]=1                    
        idx=np.flatnonzero(imgtrans==1)
        if len(list(idx))>0:
            location=np.unravel_index(idx, labels.shape)   
            rr, cc = polygon(location[0], location[1])
            imgtrans[rr, cc]=1
            if np.abs(np.sum(imgtrans)-np.sum(imgt))/np.sum(imgt)<0.1:
                label_adjusted[rr, cc]=elec      
    return label_adjusted, boundary_segments_1      
        
def fit(data_x, data_y):
    #拟合直线函数
    #输入为拟合点的x与y值data_x, data_y
    #输出为斜率w和截矩b
    m = len(data_y)
    x_bar = np.mean(data_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    if sum_x2 - m * (x_bar ** 2)!=0:
        w = sum_yx / (sum_x2 - m * (x_bar ** 2))
    else:
        w=1e10 #截矩近似无限大

    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_delta += (y - w * x)
    b = sum_delta / m
    return w, b
    
def find_neighbours_along_a_way(x, y, skeleton, radius):
    se=np.zeros((skeleton.shape[0]+2, skeleton.shape[1]+2), dtype='u1')
    se[1:-1,1:-1]=skeleton
    x0=x+1
    y0=y+1
    path=[[x0,y0]]
    xc=x0
    yc=y0
    for i in range(radius):
        patch=se[xc-1:xc+2, yc-1:yc+2]
        if np.sum(patch)<4:
            if se[xc-1][yc-1]==1:
                if [xc-1, yc-1] not in path:
                    xc=xc-1
                    yc=yc-1
                    path.append([xc,yc])
                    continue
            if se[xc-1][yc]==1:
                if [xc-1,yc] not in path:
                    xc=xc-1
                    yc=yc
                    path.append([xc,yc])
                    continue                    
            if se[xc-1][yc+1]==1:
                if [xc-1,yc+1] not in path:
                    xc=xc-1
                    yc=yc+1
                    path.append([xc,yc])
                    continue                    
            if se[xc][yc-1]==1:
                if [xc,yc-1] not in path:
                    xc=xc
                    yc=yc-1
                    path.append([xc,yc]) 
                    continue                     
            if se[xc][yc+1]==1:
                if [xc,yc+1] not in path:
                    xc=xc
                    yc=yc+1
                    path.append([xc,yc]) 
                    continue                   
            if se[xc+1][yc-1]==1:
                if [xc+1,yc-1] not in path:
                    xc=xc+1
                    yc=yc-1
                    path.append([xc,yc])
                    continue                    
            if se[xc+1][yc]==1:
                if [xc+1,yc] not in path:
                    xc=xc+1
                    yc=yc
                    path.append([xc,yc])
                    continue                    
            if se[xc+1][yc+1]==1:
                if [xc+1,yc+1] not in path:
                    xc=xc+1
                    yc=yc+1
                    path.append([xc,yc])
                    continue                    
        else:
            break

    row0, cols0=skeleton.shape
    path_idx=[]
    for p in path:
        x=p[0]
        y=p[1]
        x0=x-1
        y0=y-1
        idx=y0+x0*cols0
        path_idx.append(idx)
    return path_idx
    
def remove_over_skeleton_pixels(skeleton, concaves):

    rows, cols=skeleton.shape
    structure=np.ones((3,3), dtype='u1')  
    skeleton_convolve=signal.convolve2d(skeleton, structure, boundary='fill', mode='same')
    skeleton_convolve=skeleton_convolve*skeleton
    skeleton_extension=np.zeros((rows+4, cols+4), dtype='u1')
    skeleton_extension[2:-2,2:-2]=skeleton_convolve    
    
    mask=np.zeros((5,5), dtype='u1')
    mask[1:4,1:4]=1
    mask[2][2]=0
    idx_mask=list(np.flatnonzero(mask==1))
        
    for p in concaves:
        x0=p//cols
        y0=p-x0*cols
        x=x0+2
        y=y0+2
        if skeleton_extension[x][y]>2:
            patch=skeleton_extension[x-2:x+3, y-2:y+3]
            patch_copy=patch.copy()
            patch_copy[2][2]=0
            patch_copy=patch_copy>0
            patch_copy=patch_copy.astype('u1')
            patch_convolve=signal.convolve2d(patch_copy, structure, boundary='fill', mode='same')
            idx=list(np.flatnonzero(patch_convolve==2))
            idx=list(set(idx).intersection(set(idx_mask)))
            if len(idx)>0:
                skeleton_extension[x][y]=0
                new_end=idx[0]
                xr=new_end//5
                yr=new_end-xr*5
                xt=x+xr-4
                yt=y+yr-4
                concaves[concaves.index(p)]=xt*cols+yt
                #print('change concaves point: ', xt, yt)
    skeleton=skeleton_extension[2:-2,2:-2]
    skeleton=skeleton>0
    skeleton=skeleton.astype('u1')

    return skeleton, concaves

def skeleton_extension(skeleton, skeleton_initial, radius, concaves):
    #该程序用于将孔隙骨架的分支端点与最近凹点相互连接
    #skeleton是当前孔隙骨架
    #concaves是凹点集合
    #输出为改造后的孔隙骨架和新的凹点集合
    rows, cols = skeleton.shape
    edt, inds = ndimage.distance_transform_edt(1-skeleton, return_indices=True) #计算图中所有点到孔隙骨架的最近距离以及对应的最近骨架点的坐标   
    #================================将每一个凹点与距其最近的骨架点连接起来===================================
    for i in range(len(concaves)): #对每一个凹点逐个分析
        inter_concave=concaves[i]
        nx=inter_concave//cols #nx, ny分别代表当前凹点的坐标
        ny=inter_concave-nx*cols
        endx=inds[0][nx][ny] #endx, endy分别代表当前凹点与离其最近的孔隙骨架的最近距离所对应的骨架点坐标
        endy=inds[1][nx][ny]
        rr, cc=line(endx, endy, nx, ny)
        skeleton[rr, cc]=1
    #=============================================凹点骨架点连接完毕=========================================  

    skeleton, concaves=remove_over_skeleton_pixels(skeleton, concaves)
    concaves1=concaves.copy() 

    extended_skeleton=[]
    
    for p in concaves1:
        imgtest=np.zeros(skeleton.shape, dtype='u1')
        nx=p//cols
        ny=p-nx*cols 
        idx=find_neighbours_along_a_way(nx, ny, skeleton, radius)

        idx=[idx[0], idx[-1]]#??????????????????????????????????????????????????????????????
        idx=np.array(idx)                
        location=np.unravel_index(idx, (rows, cols))
        k, b = fit(location[0], location[1])     
        endx=idx[-1]//cols
        endy=idx[-1]-endx*cols 
        
        if nx==endx:
            x=nx
            if ny<endy:
                y=ny-radius
            if ny>endy:
                y=ny+radius 
            if ny==endy:
                y=ny                            
        if nx<endx:
            x=nx-radius/(np.sqrt(1+k**2))
            if x<0:
                x=0
            y=k*x+b              
        if nx>endx:
            x=nx+radius/(np.sqrt(1+k**2))
            if x>rows-1:
                x=rows-1                
            y=k*x+b 
             
        if y<0:
            y=0
        if y>cols-1:
            y=cols-1  
             
        x=int(np.round(x))
        y=int(np.round(y)) 
 
        concaves1[concaves1.index(p)]=x*cols+y               
        rr, cc=line(nx, ny, x, y)        
        imgtest[rr, cc]=1
        idx_imgtest=np.flatnonzero(imgtest==1)
        extended_skeleton.append(list(idx_imgtest))
    
    cross_skeleton=[]    
    for p1 in extended_skeleton:
        for p2 in extended_skeleton:
            if p1!=p2:
                uni=list(set(p1).intersection(set(p2)))
                if len(uni)>0:
                    cross_skeleton.append(p1)
                    cross_skeleton.append(p2) 
    for p in extended_skeleton:
        if p not in cross_skeleton:
            skeleton.flat[p]=1                                                           
    
    for p in concaves1:
        if skeleton.flat[p]==0:
            concaves1[concaves1.index(p)]=concaves[concaves1.index(p)]

    return skeleton, concaves1



def Remove_diagonal_coneected_pore_pixel(img):
    img_extend=np.ones((img.shape[0]+2, img.shape[1]+2), dtype=img.dtype)
    img_extend[1:-1, 1:-1]=img
    rows, cols=img_extend.shape
    img_dilation=morphology.binary_dilation(img_extend).astype(img_extend.dtype)
    diff=img_dilation-img_extend
    idx=np.flatnonzero(diff==1)
    for p in list(idx):
        x=p//cols
        y=p-x*cols
        if img_extend[x][y-1]+img_extend[x][y+1]+img_extend[x-1][y]+img_extend[x+1][y]==4:
            img_extend[x][y]=1
    img=img_extend[1:-1, 1:-1]
    return img

def fill_fragmentary_grains(img, fragments, contour_points, convex_erro):
    #该函数主要是将分割中产生的碎片重新归并，这些碎片主要表现为边界平直且都不与孔隙接触
    #img是一个二维矩阵，存储的是已经分割好的颗粒
    #fragments是一个二维矩阵，存储的是还没有进行有效分割的颗粒或是分割残留的碎片
    #contour_points是所有边界点的集合
    #convex_erro是一个小数，表示分割出的颗粒与其最小凸多边形的差距比例，越小说明颗粒越接近凸多边形
    #返回值是修改后的img和fragments

    rows, cols=img.shape
    
    fragments_t=Remove_diagonal_coneected_pore_pixel(1-fragments)
    remain_points=fragments-(1-fragments_t)
    idx_remain_points=np.flatnonzero(remain_points==1)
    edt, inds = ndimage.distance_transform_edt(remain_points, return_indices=True) 
    img.flat[idx_remain_points]=img[inds[0].flat[idx_remain_points], inds[1].flat[idx_remain_points]]
    fragments=1-fragments_t
    
    labels, num = ndimage.label(fragments, np.ones((3,3), dtype='u1'))
    ele=list(np.unique(labels))
    if len(ele)>1:
        ele=ele[1:]
        print('isolated fragments are: ', ele)
        for elec in ele:
            obj=labels==elec
            obj=obj.astype('u1')
            idx=np.flatnonzero(obj==1)
            obj_erode=morphology.binary_erosion(obj)
            boundary=obj-obj_erode
            idx_boundary=list(np.flatnonzero(boundary==1))
            inter=list(set(idx_boundary).intersection(set(contour_points)))
            if float(len(inter))/float(len(list(idx_boundary)))<0.1:
                obj_extension=morphology.binary_dilation(obj)
                diff=obj_extension-obj
                idx_neighbour=np.flatnonzero(diff==1)
                neighbours=img.flat[idx_neighbour]
                neighbours=list(set(list(neighbours)))
                if 0 in neighbours:
                    neighbours.remove(0)  
                merge=obj.copy()
                print('current objects neighbours are:', neighbours)
                for neig in neighbours:
                    neig_obj=img==neig
                    merge_two=neig_obj+obj
                    chull = convex_hull_image(merge_two)
                    chull=chull.astype('u1')
                    diff_convex=chull-merge_two
                    print('convex_error is: ', np.sum(diff_convex)/float(np.sum(merge_two)))
                    if np.sum(diff_convex)/float(np.sum(merge_two))<convex_erro:
                        img.flat[idx]=neig
                        idx=np.flatnonzero(merge_two==1)
                        fragments.flat[idx]=0
                        obj=merge_two
    return img, fragments
                    
def Grain_partition_skeleton_extension_2d(img, target_phase, its_max, threshold_size, extension_size, convex_erro):

    #img是一个segmented image
    #target_phase是一个正整数，例如1，代表目标相
    #its_max是最大迭代次数，例如200次，次迭代次数代表一直使用侵蚀算法直到所有的颗粒核心都被分辨出来
    #threshold_size是一个正整数，例如500，代表最小颗粒大小，对于面积小于500像素点的颗粒将被认为是噪音被消除
    #extension_size是指骨架入侵深度，比如extension_size=5代表每次入侵深度是5个像素点
    #输出结果是颗粒分割好的图像
    
    img=img==target_phase
    img=np.array(img).astype('u1')
    target_phase=1 #此时target_phase已经被调整为1，背景相标为0
    img=Remove_diagonal_coneected_pore_pixel(img) #去除以4-联通结构与孔隙主题连接的单个孔隙，该种类型孔隙影响cv2.findContours函数  
    img0=img.copy() 
    img_initial=img.copy()
    
    dimension=len(list(img.shape))
    if dimension==2:
        structure=morphology.square(3)
    else:
        structure=morphology.ball(1)

    #==============================smooth the boundary===========================================
    #img=smooth_image(img)   
     
    #============================================================================================
    concaves, contour_points, dominant_points=concave_points(img, 2.5) #计算关键点
    segments_all=contour_segments(img, concaves) #计算所有的边界片段
    #========================establish multiscale grain partition labels=========================
    img_partition=np.zeros(img.shape, dtype='u4') #建立一个空的区域，方便后面将每一个识别出来的凸多边形颗粒填入其中

    grain=1 #初始颗粒编号赋为1    
    finish=0 #若为0则继续分割，若为1则分割结束
    i=1 #记录这是第几次分割
    while (finish==0):
    
        label=Grain_Partition_Erosion_Dilation(img, target_phase, 500) #初始分割
        
        #=============由于骨架扩展侵蚀，label中存在一些缺失的部分（侵蚀部分）=================
        
        label_solid_idx=np.flatnonzero(label>0)
        img_initial_solid_idx=np.flatnonzero(img_initial>0)
        idx_eroded=list(set(list(img_initial_solid_idx)).difference(set(list(label_solid_idx))))
        if len(idx_eroded)>0:
            idx_eroded=np.array(idx_eroded)
            img_eroded=np.zeros(img.shape, dtype='u1')
            img_eroded.flat[idx_eroded]=1
            edt, inds = ndimage.distance_transform_edt(img_eroded, return_indices=True)
            label.flat[idx_eroded]=label[inds[0].flat[idx_eroded], inds[1].flat[idx_eroded]]

        #=======================================侵蚀部分补充完毕====================================
 
        
        #===========================判断当前分割方案下所有颗粒的凹凸性================================
        current_img=label.copy()
        obj=np.unique(current_img)
        obj=list(obj[1:])
        
        for k in obj:
            idx_obj=np.flatnonzero(current_img==k)
            current_obj=np.zeros(current_img.shape, dtype='u1')
            current_obj.flat[idx_obj]=1
            boundary_segments=boundary_segment(current_obj, contour_points)
            imgtest, boundary_segments_1 = adjust_partitioning(current_obj, segments_all) 
                          
            chull = convex_hull_image(imgtest)
            diff_convex=chull-imgtest
            diff_num=float(np.sum(diff_convex))
            imgtest_num=float((np.sum(imgtest)))
            if imgtest_num==0:
                continue
            else:
                convex_error1=diff_num/imgtest_num
            
            print('The obj_num is'+str(k)+' and its convex_error is: ' + str(diff_num)+'/'+str(imgtest_num)+'='+str(convex_error1))

            flag=[]
            for boundary in boundary_segments:
                if len(boundary)>8: #仅仅分析稍微长一点的片段，避免受噪声影响
                    intersect=list(set(boundary[4:-4]).intersection(set(list(concaves)))) #求该片段中存在的凹点，
                    #注意由于图像边界是离散信号，存在凹点与片段端点略微偏差的情形，所以取的是boundary[4:-4]，而不是boundary      
                    if len(intersect)==0:
                        flag.append(0)
                    else:
                        flag.append(1)
            if sum(flag)==0 or convex_error1<convex_erro: #若是颗粒表面没有边界或是颗粒已经是凸多边形要求（因为有些很微小的凹点存在）
                idx_junction=np.flatnonzero(imgtest>0)
                img_partition.flat[idx_junction]=grain
                grain=grain+1
                #去除用过的concaves点
                for p in boundary_segments_1:
                    if p[0] in concaves:
                        concaves.remove(p[0]) 
                    if p[-1] in concaves:
                        concaves.remove(p[-1]) 
        #===================凹凸性判断完毕，并将所有凸颗粒装入img_partition之中============================
        
        #img_partition, boundary_segments_1=adjust_partitioning(img_partition, segments_all) #调整颗粒边界分界点位置
        
        img_partition, img = fill_fragmentary_grains(img_partition, img, contour_points, convex_erro)        
        
        #========================================将分割好的颗粒从img中去除================================
        idx_img_partition_0=np.flatnonzero(img_partition>0)
        img.flat[idx_img_partition_0]=0
        img_initial.flat[idx_img_partition_0]=0
        #=========================================颗粒去除完毕============================================        

        #=======================================skeleton_extension========================================        
        img=img_initial.copy()             
        skeleton = skeletonize(1-img)
        skeleton_refer=skeleton.copy()

        concaves1=concaves.copy()        
        for t in range(i):
            skeleton, concaves1=skeleton_extension(skeleton, skeleton_refer, extension_size, concaves1)  

        idx0=np.flatnonzero(skeleton==1)                        
        img.flat[idx0]=0
    
        if np.sum(img)==0: #如果所有颗粒都已经被分割识别，则结束分割
            finish=1  
        i=i+1        
        if i==its_max: #如果迭代次数达到设定的最大次数，则结束分割
            finish=1
            idx_ele=np.flatnonzero(label==0)
            label=label+grain
            label.flat[idx_ele]=0
            img_partition=img_partition+label            
    print('it runs '+str(i)+'th iteration')     
    
    return img_partition
    
def particles_separate(img_path):
    img=pylib.open_image(path) 
    img=img>100 #*******************************************************************
    img=img.astype('u1') 
    img=img[:,:,0] 
    plt.imshow(img)
    plt.show()
    img=Remove_small_objects(img, None, 300, 2)
    img=img[:, :-5]
    img=Remove_diagonal_coneected_pore_pixel(img)
    label=Grain_Partition_Erosion_Dilation(img, 1, 500)     
    concaves, contour_points, dominant_points=concave_points(img, 2.5) #计算关键点  

    labels=Grain_partition_skeleton_extension_2d(img, 1, 10, 500, 5, 0.02) 
    segments_all=contour_segments(img, concaves)

    img_contours, grain_properties=contour_estimation(labels, segments_all)
    
    dirname=os.path.dirname(path)
    filename=os.path.split(path)[1]
    filename_nonsuffix=os.path.splitext(filename)[0]
    savepath=dirname+'/'+filename_nonsuffix+'_grain_properties.txt'
    np.savetxt(savepath, grain_properties)
    
    
    plt.imshow(img_contours*150+img*100)
    plt.show()
        
if __name__=='__main__':

    #----------------------------------岩石颗粒分割------------------------------
    path='~/demo1.png' #iuput the image path   
    img=open_image(path) 
    print('image shape is, ', img.shape) 
    img=img>100 #*******************************************************************
    img=img.astype('u1') 
    img=img[:,:,0] 
    img=Remove_small_objects(img, None, 300, 2)
    img=img[:, :-5]
    img=Remove_diagonal_coneected_pore_pixel(img)
    #----------------------------------------------------------------------------------

    
    label=Grain_Partition_Erosion_Dilation(img, 1, 500)

    concaves, contour_points, dominant_points=concave_points(img, 2.5) #计算关键点  

    labels=Grain_partition_skeleton_extension_2d(img, 1, 10, 500, 5, 0.02) 
    
    segments_all=contour_segments(img, concaves)

    img_contours, grain_properties=contour_estimation(labels, segments_all)
    
    dirname=os.path.dirname(path)
    filename=os.path.split(path)[1]
    filename_nonsuffix=os.path.splitext(filename)[0]
    savepath=dirname+'/'+filename_nonsuffix+'_grain_properties.txt'
    np.savetxt(savepath, grain_properties)
        
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True) 
    img_group=[img, labels, img_contours*150+img*100] 
    title=['Initial', 'Grain Partitioning', 'Grain Restoration']  
    m=0
    for ax in axes.flat:
        img=ax.imshow(img_group[m], vmin=np.min(img_group[m]), vmax=np.max(img_group[m]))
        ax.set_title(title[m])
        m=m+1
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

