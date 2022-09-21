#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import copy
import time
from rovi_utils import tflib

Param={
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

def learn(datArray,prm):
  global modPcArray,Param
  Param.update(prm)
  modPcArray=[]
  for dat in datArray:
    pc=fromNumpy(dat)
    modPcArray.append(pc)
  return modPcArray

def solve(datArray,prm):
  global scnPcArray,Param,score
  Param.update(prm)
  scnPcArray=[]
  t1=time.time()
  for dat in datArray:
    pc=fromNumpy(dat)
    scnPcArray.append(pc)
  tfeat=time.time()-t1
  print("time for calc feature",tfeat)
  t1=time.time()
  if 'tf' in Param:
    RT=tflib.toRT(tflib.dict2tf(Param['tf']))
  else:
    RT=np.eye(4)
  print("#### evaluate solver RT=", RT)
  print("#### evaluate solver repeat=", Param["repeat"])
  print("#### evaluate solver len=", len(score["transform"]))
  if Param["repeat"]!=len(score["transform"]):
    n=Param["repeat"]
    score={"transform":[RT]*n,"fitness":[None]*n,"rmse":[None]*n}
  print("#### evaluate solver eval_threshold=", Param["eval_threshold"])
#  print("#### evaluate solver transform=", score["transform"])
  for n in range(Param["repeat"]):
#  result=o3d.pipelines.registration.evaluate_registration(modPcArray[0],scnPcArray[0],Param["eval_threshold"],score["transform"][n])
    result=o3d.pipelines.registration.evaluate_registration(modPcArray[0],scnPcArray[0],Param["eval_threshold"],RT)
    score["transform"][n]=RT
    score["fitness"][n]=result.fitness
    score["rmse"][n]=result.inlier_rmse
  tmatch=time.time()-t1
  print("time for feature matching",tmatch)
  score["tfeat"]=tfeat
  score["tmatch"]=tmatch
  print("#### evaluate solver result=", score)
  return score

