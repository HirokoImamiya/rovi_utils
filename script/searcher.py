#!/usr/bin/env python3

import cv2
import numpy as np
import math
import roslib
import rospy
import tf
import tf2_ros
import open3d as o3d
import copy
import os
import sys
import yaml
from rovi.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from rovi_utils import tflib
from rovi_utils import sym_solver as rotsym
from rovi_utils import axis_solver as rotjour
from rovi_utils import axis_bb_solver as axisbb
from scipy import optimize

Param={
  "normal_radius":0.003,
  "feature_radius":0.01,
  "normal_min_nn":25,
  "distance_threshold":0.1,
  "icp_threshold":0.001,
  "rotate":0,
  "repeat":1,
  "cutter":{"base":0,"offset":0,"width":0,"crop":0}
}
Config={
  "proc":0,
  "path":"recipe",
  "scenes":["surface"],
  "solver":"o3d_solver",
  "scene_frame_ids":[],
  "master_frame_ids":[],
  "base_frame_id":"world",
  "align_frame_id":"camera/capture0"}
Score={
  "proc":[],
  "Tx":[],
  "Ty":[],
  "Tz":[],
  "Qx":[],
  "Qy":[],
  "Qz":[],
  "Qw":[]}

def P0():
  return np.array([]).reshape((-1,3))

def np2F(d):  #numpy to Floats
  f=Floats()
  f.data=np.ravel(d)
  return f

def getRT(base,ref):
  try:
    ts=tfBuffer.lookup_transform(base,ref,rospy.Time())
    rospy.loginfo("cropper::getRT::TF lookup success "+base+"->"+ref)
    RT=tflib.toRT(ts.transform)
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    RT=None
  return RT

def learn_feat(mod,param):
  pcd=solver.learn(mod,param)
  if Config["proc"]==0: o3d.io.write_point_cloud("/tmp/model.ply",pcd[0])
  return pcd

# マスター適合度（pvar）の計算
def calc_pvar():
  bins=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
  ft=np.array(solver.modFtArray[0][1].data) #ソルバー内の法線特徴量をいただく
  ft0=ft[:11] #33次元⇒11次元ごとに切り出す
  ft1=ft[11:22]
  ft2=ft[22:33]
  ftsum=ft0+ft1+ft2 #11次元に足し合わせる
  n=np.sum(ftsum,axis=0) #特徴点毎の法線数
  s2=(bins*bins).dot(ftsum) #特徴点毎の分散を求める
  s1=np.sqrt(s2/n) #σにする
  ss=np.sort(s1)[::-1] #σの大きいもの順に特徴点を並べ替え
  pvar=np.mean(ss[int(len(ss)*0.1):int(len(ss)*0.2)]) #全特徴点の大きい方から10～20%の平均σを指標にする
  print('*** pvar = ',pvar,'\n')
  return pvar

def learn_rot(pc,num,thres):
  global RotAxis,tfReg
  RotAxis=None
  if num>1:
    RotAxis=rotsym.solve(pc,num,thres)
    if len(RotAxis)>1:
      tf=TransformStamped()
      tf.header.stamp=rospy.Time.now()
      tf.header.frame_id=Config["master_frame_ids"][0]
      tf.child_frame_id=Config["master_frame_ids"][0]+'/axis'
      tf.transform=tflib.fromRT(RotAxis[0])
      tfReg.append(tf)
    else:
      RotAxis=None
      print('No axis')
      pub_err.publish("searcher::No axis")

def learn_journal(pc,base,ofs,wid):
  global JourAxis,tfReg
  JourAxis=None
  if wid>0:
    JourAxis=rotjour.solve(pc,base,base+ofs,wid)
    if JourAxis is not None:
      tf=TransformStamped()
      tf.header.stamp=rospy.Time.now()
      tf.header.frame_id=Config["master_frame_ids"][0]
      tf.child_frame_id=Config["master_frame_ids"][0]+'/journal'
      tf.transform=tflib.fromRT(JourAxis)
      tfReg.append(tf)
    else:
      print('No journal')
      pub_err.publish("searcher::No journal")

def learn_axis(pc):
  if Config["proc"]==0:
    axis=axisbb.solve(pc)
    tf=TransformStamped()
    tf.header.stamp=rospy.Time.now()
    tf.header.frame_id=Config["master_frame_ids"][0]
    tf.child_frame_id=Config["master_frame_ids"][0]+'/axis0'
    tf.transform=tflib.fromRT(axis)
    tfReg.append(tf)

def cb_master(event):
  if Config["proc"]==0:
    for n,l in enumerate(Config["scenes"]):
      print("publish master",len(Model[n]))
      if Model[n] is not None: pub_pcs[n].publish(np2F(Model[n]))

def cb_save(msg):
  global Model,tfReg
#save point cloud
  for n,l in enumerate(Config["scenes"]):
    if Scene[n] is None: continue
    pc=o3d.geometry.PointCloud()
    m=Scene[n]
    if(len(m)==0):
      pub_err.publish("searcher::save::point cloud ["+l+"] has no point")
      pub_saved.publish(mFalse)
      return
    Model[n]=m
    pc.points=o3d.utility.Vector3dVector(m)
    o3d.io.write_point_cloud(Config["path"]+"/"+l+".ply",pc,True,False)
    pub_pcs[n].publish(np2F(m))
  tfReg=[]
#copy TF scene...->master... and save them
  for s,m in zip(Config["scene_frame_ids"],Config["master_frame_ids"]):
    try:
      tf=tfBuffer.lookup_transform(Config["base_frame_id"],s,rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      tf=TransformStamped()
      tf.header.stamp=rospy.Time.now()
      tf.header.frame_id=Config["base_frame_id"]
      tf.transform.rotation.w=1
    path=Config["path"]+"/"+m.replace('/','_')+".yaml"
    f=open(path,"w")
    f.write(yaml.dump(tflib.tf2dict(tf.transform)))
    f.close()
    tf.child_frame_id=m
    tfReg.append(tf)
  if Config["proc"]==0: broadcaster.sendTransform(tfReg)
  pcd=learn_feat(Model,Param)
  pvar=calc_pvar()
  pub_pvar.publish(pvar)
  learn_rot(pcd[0],Param['rotate'],Param['icp_threshold'])
  pub_msg.publish("searcher::master plys and frames saved")
  pub_saved.publish(mTrue)
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)

def cb_load(msg):
  global Model,tfReg,Param
#load point cloud
  for n,l in enumerate(Config["scenes"]):
    pcd=o3d.io.read_point_cloud(Config["path"]+"/"+l+".ply")
    Model[n]=np.reshape(np.asarray(pcd.points),(-1,3))
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)
  tfReg=[]
#load TF such as master/camera...
  for m in Config["master_frame_ids"]:
    path=Config["path"]+"/"+m.replace('/','_')+".yaml"
    try:
      f=open(path, "r+")
    except Exception:
      pub_msg.publish("searcher error::master TF file load failed"+path)
      tf=TransformStamped()
      tf.header.stamp=rospy.Time.now()
      tf.header.frame_id=Config["base_frame_id"]
      tf.child_frame_id=m
      tf.transform.rotation.w=1
      tfReg.append(tf)
    else:
      yd=yaml.load(f,Loader=yaml.SafeLoader)
      f.close()
      trf=tflib.dict2tf(yd)
      tf=TransformStamped()
      tf.header.stamp=rospy.Time.now()
      tf.header.frame_id=Config["base_frame_id"]
      tf.child_frame_id=m
      tf.transform=trf
      tfReg.append(tf)
      Param['transform']=tflib.toRT(trf)
  Param.update(rospy.get_param("~param"))
  print('learning pc',Param['rotate'])
  pcd=learn_feat(Model,Param)
  learn_rot(pcd[0],Param['rotate'],Param['icp_threshold'])
  learn_journal(pcd[0],Param["cutter"]["base"],Param["cutter"]["offset"],Param["cutter"]["width"])
  learn_axis(pcd[0])
  if JourAxis is not None:
    lvar=Param["cutter"]
    pcx=pcd[0].transform(JourAxis)
    if lvar["crop"]>0:  #crop outside
      if lvar["offset"]>0:
        cx1=lvar["base"]+lvar["crop"]
        cx2=lvar["base"]+lvar["offset"]-lvar["crop"]
      else:
        cx1=lvar["base"]+lvar["offset"]+lvar["crop"]
        cx2=lvar["base"]-lvar["crop"]
      box1=o3d.geometry.AxisAlignedBoundingBox(min_bound=(-10000,-10000,-10000), max_bound=(cx1,10000,10000))
      pcx1=pcx.crop(box1)
      pcd1=pcx1.transform(np.linalg.inv(JourAxis))
      box2=o3d.geometry.AxisAlignedBoundingBox(min_bound=(cx2,-10000,-10000), max_bound=(10000,10000,10000))
      pcx2=pcx.crop(box2)
      pcd2=pcx2.transform(np.linalg.inv(JourAxis))
      Model[0]=np.concatenate((np.array(pcd1.points),np.array(pcd2.points)),axis=0)
    elif lvar["crop"]<0:  #crop inside
      pass
    learn_feat(Model,Param)
    rospy.Timer(rospy.Duration(0.2),cb_master,oneshot=True)
  if Config["proc"]==0: broadcaster.sendTransform(tfReg)
  pub_msg.publish("searcher::model loaded and learning completed")
  pub_loaded.publish(mTrue)

def cb_score():
  global Score
  score=Float32MultiArray()
  score.layout.data_offset=0
  for n,sc in enumerate(Score):
    score.layout.dim.append(MultiArrayDimension())
    score.layout.dim[n].label=sc
    score.layout.dim[n].size=len(Score[sc])
    score.layout.dim[n].stride=1
    score.data.extend(Score[sc])
  pub_score.publish(score)
  pub_Y2.publish(mTrue)

def cb_solve(msg):
  global Score
  if len(list(filter(lambda x:len(x)>0,Scene)))==0:
    pub_msg.publish("searcher::Lacked scene to solve")
    pub_Y2.publish(mFalse)
    return
  Param.update(rospy.get_param("~param"))
  for key in Score: Score[key]=[]
  cb_busy(mTrue)
  rospy.Timer(rospy.Duration(0.01),cb_solve_do,oneshot=True)

def cb_solve_do(msg):
  global Score
  result=solver.solve(Scene,Param)
  RTs=result["transform"]
  if np.all(RTs[0]):
    pub_err.publish("solver error")
    pub_Y2.publish(mFalse)
    return
  else:
    pub_msg.publish("searcher::"+str(len(RTs))+" model searched")

  cTb=getRT('camera/capture0','base')
  xvec=cTb[:3,:3].T[0]
  for n,rt in enumerate(RTs):
    print('Solve',tflib.fromRT(rt))
    tf=Transform()
    if RotAxis is not None:
      wrt=[]
      rot=[]
      for m,wt in enumerate(RotAxis): #to minimumize the rotation
        if m==0: wrt.append(rt)
        else: wrt.append(np.dot(rt,wt))
        R=wrt[n][:3,:3]
        rot.append(np.inner(R.T[0],xvec))
      tf=tflib.fromRT(wrt[np.argmin(np.array(rot))])
      print('TF axis 1',tf)
      solver.score["transform"][n]=tflib.toRT(tf)
      prm=Param.copy()
      prm["distance_threshold"]=0
      prm["repeat"]=1
      result=solver.solve(Scene,prm)
      tf=tflib.fromRT(result["transform"][0])
      print('TF axis 2',tf)
    else:
      tf=tflib.fromRT(rt)

    Score["Tx"].append(tf.translation.x)
    Score["Ty"].append(tf.translation.y)
    Score["Tz"].append(tf.translation.z)
    Score["Qx"].append(tf.rotation.x)
    Score["Qy"].append(tf.rotation.y)
    Score["Qz"].append(tf.rotation.z)
    Score["Qw"].append(tf.rotation.w)

  result["proc"]=float(Config["proc"])
  for key in result:
    if type(result[key]) is not list: # scalar->list
      Score[key]=[result[key]]*len(RTs)
    elif type(result[key][0]) is float: # float->list
      Score[key]=result[key]
  cb_score()

def cb_ps(msg,n):
  global Scene
  pc=np.reshape(msg.data,(-1,3))
  Scene[n]=pc
  print("cb_ps",pc.shape)

def cb_clear(msg):
  global Scene
  for n,l in enumerate(Config["scenes"]):
    Scene[n]=None
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)

def cb_busy(event):
  global Score
  if len(Score["proc"])<Param["repeat"]:
    pub_busy.publish(mTrue)
    rospy.Timer(rospy.Duration(0.5),cb_busy,oneshot=True)
  else:
    pub_busy.publish(mFalse)

def cb_dump(msg):
#dump informations
  for n,l in enumerate(Config["scenes"]):
    if Scene[n] is None: continue
    pc=o3d.geometry.PointCloud()
    m=Scene[n]
    if(len(m)==0): continue
    pc.points=o3d.utility.Vector3dVector(m)
    o3d.io.write_point_cloud("/tmp/"+l+".ply",pc,True,False)

def cb_param(msg):
  global Param
  prm=Param.copy()
  try:
    Param.update(rospy.get_param("~param"))
  except Exception as e:
    print("get_param exception:",e.args)
  if prm!=Param:
    print("Param changed",Param)
    learn_feat(Model,Param)
  rospy.Timer(rospy.Duration(1),cb_param,oneshot=True)
  return

def parse_argv(argv):
  args={}
  for arg in argv:
    tokens = arg.split(":=")
    if len(tokens) == 2:
      key = tokens[0]
      args[key] = tokens[1]
  return args

########################################################

rospy.init_node("searcher",anonymous=True)
Config.update(parse_argv(sys.argv))
try:
  Config.update(rospy.get_param("~config"))
except Exception as e:
  print("get_param exception:",e.args)
print("Config",Config)
try:
  Param.update(rospy.get_param("~param"))
except Exception as e:
  print("get_param exception:",e.args)
print("Param",Param)

###load solver
exec("from rovi_utils import "+Config["solver"]+" as solver")

###I/O
pub_pcs=[]
for n,c in enumerate(Config["scenes"]):
  rospy.Subscriber("~in/"+c+"/floats",numpy_msg(Floats),cb_ps,n)
  pub_pcs.append(rospy.Publisher("~master/"+c+"/floats",numpy_msg(Floats),queue_size=1))
pub_Y2=rospy.Publisher("~solved",Bool,queue_size=1)
pub_busy=rospy.Publisher("~stat",Bool,queue_size=1)
pub_saved=rospy.Publisher("~saved",Bool,queue_size=1)
pub_loaded=rospy.Publisher("~loaded",Bool,queue_size=1)
pub_score=rospy.Publisher("~score",Float32MultiArray,queue_size=1)
rospy.Subscriber("~clear",Bool,cb_clear)
rospy.Subscriber("~solve",Bool,cb_solve)
if Config["proc"]==0: rospy.Subscriber("~save",Bool,cb_save)
rospy.Subscriber("~load",Bool,cb_load)
if Config["proc"]==0: rospy.Subscriber("~redraw",Bool,cb_master)
if Config["proc"]==0: rospy.Subscriber("/searcher/dump",Bool,cb_dump)
pub_hash=rospy.Publisher("~hash",Int64,queue_size=1)
pub_msg=rospy.Publisher("/message",String,queue_size=1)
pub_err=rospy.Publisher("/error",String,queue_size=1)
pub_pvar=rospy.Publisher("~pvar",Float64,queue_size=1)

###std_msgs/Bool
mTrue=Bool()
mTrue.data=True
mFalse=Bool()
mFalse.data=False

###TF
tfBuffer=tf2_ros.Buffer()
listener=tf2_ros.TransformListener(tfBuffer)
broadcaster=tf2_ros.StaticTransformBroadcaster()

###data
Scene=[None]*len(Config["scenes"])
Model=[None]*len(Config["scenes"])
RotAxis=None
JourAxis=None
tfReg=[]

rospy.Timer(rospy.Duration(5),cb_load,oneshot=True)
rospy.Timer(rospy.Duration(1),cb_param,oneshot=True)
try:
  rospy.spin()
except KeyboardInterrupt:
  print("Shutting down")
