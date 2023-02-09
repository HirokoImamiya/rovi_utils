#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import copy
import time
from rovi_utils import tflib

Param={
  "normal_radius":0.01,
  "normal_min_nn":0,
  "icp_threshold":0.003,
  "eval_threshold":0
}

modPcArray=[]
scnPcArray=[]
score={"transform":[np.eye(4)],"fitness":[None],"rmse":[None]}

def toNumpy(pcd):
  return np.reshape(np.asarray(pcd.points),(-1,3))

def fromNumpy(dat):
  pc=o3d.geometry.PointCloud()
  pc.points=o3d.utility.Vector3dVector(dat)
  return pc

def _get_features(cloud):
  o3d.geometry.PointCloud.estimate_normals(cloud,o3d.geometry.KDTreeSearchParamRadius(radius=Param["normal_radius"]))
  viewpoint=np.array([0.0,0.0,0.0],dtype=float)
  o3d.geometry.PointCloud.orient_normals_towards_camera_location(cloud, camera_location=viewpoint)
  nfmin=Param["normal_min_nn"]
  if nfmin<=0: nfmin=1
  cl,ind=o3d.geometry.PointCloud.remove_radius_outlier(cloud,nb_points=nfmin,radius=Param["normal_radius"])
  nfcl=o3d.geometry.PointCloud.select_by_index(cloud,ind)
  cloud.points=nfcl.points
  cloud.normals=nfcl.normals
  cds=cloud
#  if Param["feature_mesh"]>0:
#    cds=o3d.geometry.PointCloud.voxel_down_sample(cloud,voxel_size=Param["feature_mesh"])
#  return cds,o3d.pipelines.registration.compute_fpfh_feature(cds,o3d.geometry.KDTreeSearchParamRadius(radius=Param["feature_radius"]))
  return cds

def learn(datArray,prm):
  global modPcArray,Param
  Param.update(prm)
  modPcArray=[]
  for dat in datArray:
    pc=fromNumpy(dat)
    modPcArray.append(pc)
    if not pc.is_empty():
      _get_features(pc)
  return modPcArray

def solve(datArray,prm):
  global scnPcArray,Param,score
  Param.update(prm)
  scnPcArray=[]
  t1=time.time()
  for dat in datArray:
    pc=fromNumpy(dat)
    scnPcArray.append(pc)
    if not pc.is_empty():
      _get_features(pc)
  tfeat=time.time()-t1
  print("evaluate solver::time for calc feature",tfeat)
  t1=time.time()
  if 'tf' in Param:
    RT=tflib.toRT(tflib.dict2tf(Param['tf']))
  else:
    RT=np.eye(4)
  if Param["repeat"]!=len(score["transform"]):
    n=Param["repeat"]
    score={"transform":[RT]*n,"fitness":[None]*n,"rmse":[None]*n}
  for n in range(Param["repeat"]):
    score["transform"][n]=RT
    if Param["icp_threshold"]>0:
      result=o3d.pipelines.registration.registration_icp(
        modPcArray[0],scnPcArray[0],
        Param["icp_threshold"],
        score["transform"][n],o3d.pipelines.registration.TransformationEstimationPointToPlane())
      score["transform"][n]=result.transformation
      score["fitness"][n]=result.fitness
      score["rmse"][n]=result.inlier_rmse
    if Param["eval_threshold"]>0:
      result=o3d.pipelines.registration.evaluate_registration(modPcArray[0],scnPcArray[0],Param["eval_threshold"],score["transform"][n])
      score["fitness"][n]=result.fitness
      score["rmse"][n]=result.inlier_rmse
  tmatch=time.time()-t1
  print("evaluate solver::time for feature matching",tmatch)
  score["tfeat"]=tfeat
  score["tmatch"]=tmatch
  print("evaluate solver::result=", score)
  return score

