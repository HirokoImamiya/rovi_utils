#!/usr/bin/env python3

import numpy as np
import rospy
import tf2_ros
import open3d as o3d
import os
import sys
import time
import json
import yaml
from scipy.spatial.transform import Rotation as R
from rovi.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from rovi_utils import tflib
from rovi_utils.srv import TextFilter
from rovi_utils import sym_solver as rotsym

Param={
  "evaluate":False,
  "pitch":0,
  "var":[],
  "pitch_subd":0,
  "repeat":1,
  "eval_threshold":0,
  "icp_threshold":0,
  "rotate":0
}
Config={
  "proc":0,
  "path":"recipe",
  "scenes":["surface_sub"],
  "solver":"evaluate_solver",
  "scene_frame_ids":[],
  "master_frame_ids":[],
  "solve_frame_id":"camera/capture0",
  "base_frame_id":"world",
  "master_main_frame_id":"camera/master0",
  "axis_frame_id":"axis",
  "debugoutput":False
}
EvalScore={
  "fitness":[None],
  "rmse":[None],
  "transform":[None]
}

def P0():
  return np.array([]).reshape((-1,3))

def np2F(d):  #numpy to Floats
  f=Floats()
  f.data=np.ravel(d)
  return f

def getRT(base,ref):
  try:
    ts=tfBuffer.lookup_transform(base,ref,rospy.Time())
    rospy.loginfo("post::getRT::TF lookup success "+base+"->"+ref)
    RT=tflib.toRT(ts.transform)
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    RT=None
  return RT

def learn_feat(mod,param):
  pcd=solver.learn(mod,param)
  if Config["proc"]==0: o3d.io.write_point_cloud("/tmp/sub_model.ply",pcd[0])
  return pcd

def learn_rot(pc,num,thres):
  global RotAxis,tfReg
  RotAxis=None
  if num>1 and (not pc.is_empty()):
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

def solve_rot(tr):
  global EvalScore,MsgScore
  for key in EvalScore: EvalScore[key]=[]
  MsgScore=list()
  return do_eval_solve(tr,rot=True)

def isEvaluate():
  return Param['evaluate']

def get_solve_result(source, target):
  ts=Transform()
  if tf_lookup is not None:
    req='base '+source+ ' '+target
    try:
      res=tf_lookup(req)
    except rospy.ServiceException as exc:
      print("Service did not process reqest: ", str(exc))
    else:
      ts=tflib.dict2tf(json.loads(res.data))
  else:
    print("tf_lookup None")
  return ts

def change_main_result():
  base=Config['base_frame_id']
  ts=get_solve_result(Config['master_main_frame_id'], Config['solve_frame_id']+'/solve0')
  bTc=getRT('base',Config["master_frame_ids"][0])
  cTb=np.linalg.inv(bTc)
  cTc=tflib.toRT(ts)
  Tm=cTb.dot(cTc).dot(bTc)
  wTc=getRT(Config["solve_frame_id"],Config["master_frame_ids"][0])
  tr=tflib.fromRT(wTc.dot(Tm))
  set_solve_sub_pos(tr)
  return tr

def set_axis_pos():
  base=Config['base_frame_id']
  if RotAxis is not None:
    target=Config["master_frame_ids"][0]+'/axis'
    solve_source=Config["master_frame_ids"][0]
    solve_target=Config['solve_frame_id']+'/solve0_sub'
  else:
    target=Config['master_main_frame_id']+'/axis'
    solve_source=Config['master_main_frame_id']
    solve_target=Config['solve_frame_id']+'/solve0'
  axis_master=Config['axis_frame_id']+'/master0'
  axis_solve=Config['axis_frame_id']+'/solve0'
  wTc=getRT(base,target)
# Config['master_main_frame_id']+'/axis' は 中心軸ソルバ(rotsym_solver)の結果
# Config['master_main_frame_id'] マスタの最初の撮影点は対象の中心軸付近（上空から撮影）を前提
  if wTc is not None:
    tf=TransformStamped()
    tf.header.stamp=rospy.Time.now()
    tf.header.frame_id=base
    tf.child_frame_id=axis_master
    tf.transform=tflib.fromRT(wTc)
    broadcaster.sendTransform(tf)

    ts=get_solve_result(solve_source,solve_target)
    bTc=getRT('base',axis_master)
    cTb=np.linalg.inv(bTc)
    cTc=tflib.toRT(ts)
    Tm=cTb.dot(cTc).dot(bTc)
    wTc=getRT(base,axis_master)
    tf=TransformStamped()
    tf.header.stamp=rospy.Time.now()
    tf.header.frame_id=base
    tf.child_frame_id=axis_solve
    tf.transform=tflib.fromRT(wTc.dot(Tm))
    broadcaster.sendTransform(tf)

def set_solve_sub_pos(tr):
  tf=TransformStamped()
  tf.header.stamp=rospy.Time.now()
  tf.header.frame_id=Config["solve_frame_id"]
  tf.child_frame_id=Config["solve_frame_id"]+"/solve0_sub"
  tf.transform=tr
  broadcaster.sendTransform(tf)

def set_score(score):
  global MsgScore,EvalScore
  msg_score=Float32MultiArray()
  msg_score.layout.data_offset=0
  for n,sc in enumerate(score):
    msg_score.layout.dim.append(MultiArrayDimension())
    msg_score.layout.dim[n].label=sc
    msg_score.layout.dim[n].size=len(score[sc])
    msg_score.layout.dim[n].stride=1
    msg_score.data.extend(score[sc])
  MsgScore.append(msg_score)
  pick=np.argmax(score['fitness'])
  EvalScore["fitness"].append(score['fitness'][pick])
  EvalScore["rmse"].append(score['rmse'][pick])
  ts=Transform()
  ts.translation.x=score["Tx"][pick]
  ts.translation.y=score["Ty"][pick]
  ts.translation.z=score["Tz"][pick]
  ts.rotation.x=score["Qx"][pick]
  ts.rotation.y=score["Qy"][pick]
  ts.rotation.z=score["Qz"][pick]
  ts.rotation.w=score["Qw"][pick]
  EvalScore["transform"].append(ts)

def cb_done(result,finish=False):
  global isExec,T1
  if finish:
    tsolve=time.time()-T1
    stats={}
    stats['tsolve']=tsolve
    pub_report.publish(str(stats))
  f=Bool()
  f.data=result
  pub_Y2.publish(f)
  isExec=False

def cb_score(n,finish):
  if finish:
    pub_score.publish(MsgScore[n])
  set_solve_sub_pos(EvalScore["transform"][n])
  if Config['debugoutput']:
    rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)

def cb_master(event):
  if Config["proc"]==0:
    for n,l in enumerate(Config["scenes"]):
      if Model[n] is not None:
        print("post::publish master",len(Model[n]))
        pub_pcs[n].publish(np2F(Model[n]))

def cb_save(msg):
  global Model,tfReg
  if isEvaluate() is False:
    pub_saved.publish(mTrue)
    return
#save point cloud
  for n,l in enumerate(Config["scenes"]):
    if Scene[n] is None: continue
    pc=o3d.geometry.PointCloud()
    m=Scene[n]
    if(len(m)==0):
      pub_err.publish("post::save::point cloud ["+l+"] has no point")
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
  learn_rot(pcd[0],Param['rotate'],Param['threshold'])
  pub_msg.publish("post::master plys and frames saved")
  pub_saved.publish(mTrue)
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)

def cb_load(msg):
  global Model,tfReg,Param
#load point cloud
  for n,l in enumerate(Config["scenes"]):
    if os.path.isfile(Config["path"]+"/"+l+".ply"):
      pcd=o3d.io.read_point_cloud(Config["path"]+"/"+l+".ply")
      Model[n]=np.reshape(np.asarray(pcd.points),(-1,3))
    else:
      Model[n]=[]
  tfReg=[]
#load TF such as master/camera...
  for m in Config["master_frame_ids"]:
    path=Config["path"]+"/"+m.replace('/','_')+".yaml"
    try:
      f=open(path, "r+")
    except Exception:
      pub_msg.publish("post error::master TF file load failed"+path)
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
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)
  update_param()
  pcd=learn_feat(Model,Param)
  learn_rot(pcd[0],Param['rotate'],Param['threshold'])
  if Config["proc"]==0: broadcaster.sendTransform(tfReg)
  pub_msg.publish("post::model loaded and learning completed")
  pub_loaded.publish(mTrue)

def cb_solve(msg):
  global isExec,T1
  isExec=True
  cb_busy(mTrue)
  T1=time.time()
  update_param()
  print("cb_solve start")

def cb_main_solved(msg):
  if msg.data is False:
    cb_done(False)

def cb_main_judged(msg):
  if isEvaluate():
    if msg.data:
      if len(list(filter(lambda x:len(x)>0,Scene)))==0:
        pub_msg.publish("post::Lacked scene to solve")
        cb_done(False)
      else:
        print("cb_solve_do start")
        rospy.Timer(rospy.Duration(0.01),cb_solve_do,oneshot=True)
    else:
      cb_done(msg.data,True)
      print("post solve main end")
  else:
    cb_done(msg.data,True)
    print("post solve main end")

def cb_sub_judged(msg):
  cb_done(msg.data,True)
  print("post solve sub end")

def do_eval_solve(tr,rot=False):
  score={"proc":[],"Tx":[],"Ty":[],"Tz":[],"Qx":[],"Qy":[],"Qz":[],"Qw":[]}
  prm=Param.copy()
  prm['tf']=tflib.tf2dict(tr)
  if rot:
    prm["eval_threshold"]=0
    prm["icp_threshold"]=prm["threshold"]
  result=solver.solve(Scene,prm)
  RTs=result["transform"]
  if np.all(RTs[0]):
    pub_err.publish("post::solver error")
    return False
  else:
    pub_msg.publish("post::"+str(len(RTs))+" model searched")

  if rot:
    cTb=getRT('camera/capture0','base')
    xvec=cTb[:3,:3].T[0]
  for n,rt in enumerate(RTs):
    print('Post',tflib.fromRT(rt))
    if rot:
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
    else:
     tf=tflib.fromRT(rt)

    score["Tx"].append(tf.translation.x)
    score["Ty"].append(tf.translation.y)
    score["Tz"].append(tf.translation.z)
    score["Qx"].append(tf.rotation.x)
    score["Qy"].append(tf.rotation.y)
    score["Qz"].append(tf.rotation.z)
    score["Qw"].append(tf.rotation.w)

  result["proc"]=float(Config["proc"])
  for key in result:
    if type(result[key]) is not list: # scalar->list
      score[key]=[result[key]]*len(RTs)
    elif type(result[key][0]) is float: # float->list
      score[key]=result[key]
  set_score(score)
  return True

def rotz_eval_solve(wTc,bTu,vars):
  global EvalScore,MsgScore
  for key in EvalScore: EvalScore[key]=[]
  MsgScore=list()
  result=False
  if wTc is not None:
    for n,rz in enumerate(vars):
      rospy.loginfo("post::eval solve n=%d degree=%.2f",n,rz)
      rt=np.eye(4)
      rt[:3,:3]=R.from_euler('z',rz,degrees = True).as_matrix()
      wTcc=rt.dot(wTc)
      euler=R.from_matrix(wTcc[:3,:3]).as_euler('xyz',degrees=True)
      pos=[wTcc[0,3],wTcc[1,3],wTcc[2,3],euler[0],euler[1],euler[2]]
      rot=R.from_euler('xyz', pos[3:6], degrees=True)
      Tm=np.eye(4)
      Tm[:3,:3]=rot.as_matrix()
      Tm[:3,3]=np.array(pos[:3]).T
      result=do_eval_solve(tflib.fromRT(bTu.dot(Tm)))
      if result is False:
        break
      if Config['debugoutput']:
        rospy.loginfo("post::eval score n=%d fitness=%.5f rmse=%.5f",n,EvalScore["fitness"][n],EvalScore["rmse"][n])
        cb_score(n,False)
        stats={}
        stats['angle_temp']=rz
        stats['fitness_temp']=EvalScore["fitness"][n]
        pub_report.publish(str(stats))
        rospy.sleep(0.01)
  return result

def cb_solve_do(msg):
  tr=change_main_result()
  if RotAxis is not None:
    if solve_rot(tr):
      cb_score(0,False)
      rospy.sleep(0.01)
    else:
      cb_done(False)
      return
  set_axis_pos()
  rospy.sleep(0.01)

  if Param['pitch']: vars=np.arange(Param['var'][0],Param['var'][1],Param['pitch'])
  else: vars=Param['var']
  uf=Config['axis_frame_id'] + '/solve0'
  source=Config['solve_frame_id']
  target=Config['solve_frame_id']+'/solve0_sub'
  wTc=getRT(uf, target)
  bTu=getRT(source, uf)
  result=rotz_eval_solve(wTc,bTu,vars)
  stats={}
  if result:
    m=np.argmax(EvalScore['fitness'])
    stats['angle_step1']=vars[m]
    stats['fitness_step1']=EvalScore["fitness"][m]
    pub_report.publish(str(stats))
    if Param['pitch'] and Param['pitch_subd']:
      start=vars[m]-Param['pitch']/2
      end=vars[m]+Param['pitch']/2
      print("post eval step2 m=", m, "range=", start, ",", end)
      vars=np.arange(start,end,Param['pitch_subd'],float)
      result=rotz_eval_solve(wTc,bTu,vars)
      if result:
        m=np.argmax(EvalScore['fitness'])
        maxindex=[i for i, x in enumerate(EvalScore['fitness']) if x == max(EvalScore['fitness'])]
        print("post result max=", maxindex)
        stats['angle_step2']=vars[m]
        stats['fitness_step2']=EvalScore["fitness"][m]

  if result:
    rospy.loginfo("post::eval score fix angle=%.3f num=%d",vars[m],m)
    cb_score(m,True)
    stats['angle']=vars[m]
  else:
    cb_done(False)

  if len(stats):
    pub_report.publish(str(stats))

def cb_ps(msg,n):
  global Scene
  pc=np.reshape(msg.data,(-1,3))
  Scene[n]=pc
  print("cb_ps",pc.shape)

def cb_clear(msg):
  global Scene
  for n,l in enumerate(Config["scenes"]):
    Scene[n]=None
  tr=Transform()
  tr.translation.x=0
  tr.translation.y=0
  tr.translation.z=1000000
  tr.rotation.x=0
  tr.rotation.y=0
  tr.rotation.z=0
  tr.rotation.w=1
  set_solve_sub_pos(tr)
  rospy.Timer(rospy.Duration(0.1),cb_master,oneshot=True)

def cb_busy(event):
  if isExec:
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
  update_param()
  if prm!=Param:
    print("Param changed",Param)
    learn_feat(Model,Param)
  rospy.Timer(rospy.Duration(1),cb_param,oneshot=True)
  return

def update_param():
  global Param
  try:
    Param.update(rospy.get_param("~param"))
  except Exception as e:
#    print("get_param exception:",e.args)
    pass
  Param['threshold']=Param['eval_threshold']
  if Param['icp_threshold']:
    Param['threshold']=Param['icp_threshold']

def parse_argv(argv):
  args={}
  for arg in argv:
    tokens = arg.split(":=")
    if len(tokens) == 2:
      key = tokens[0]
      args[key] = tokens[1]
  return args

########################################################

rospy.init_node("post",anonymous=True)
Config.update(parse_argv(sys.argv))
try:
  Config.update(rospy.get_param("~config"))
except Exception as e:
  print("get_param exception:",e.args)
print("Config",Config)
update_param()
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
rospy.Subscriber("~main_solved",Bool,cb_main_solved)
rospy.Subscriber("~main_judged",Bool,cb_main_judged)
rospy.Subscriber("~sub_judged",Bool,cb_sub_judged)
if Config["proc"]==0: rospy.Subscriber("~save",Bool,cb_save)
rospy.Subscriber("~load",Bool,cb_load)
if Config["proc"]==0: rospy.Subscriber("~redraw",Bool,cb_master)
if Config["proc"]==0: rospy.Subscriber("~dump",Bool,cb_dump)
pub_msg=rospy.Publisher("/message",String,queue_size=1)
pub_err=rospy.Publisher("/error",String,queue_size=1)
pub_report=rospy.Publisher('/report',String,queue_size=1)

try:
  rospy.wait_for_service('/tf_lookup/query', 2000)
except rospy.ROSInterruptException as e:
  print("tf_lookup service not shutdown")
  tf_lookup = None
except rospy.ROSException as e:
  print("tf_lookup service not available")
  tf_lookup = None
else:
  tf_lookup = rospy.ServiceProxy('/tf_lookup/query', TextFilter)

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
tfReg=[]
MsgScore=list()
isExec=False

rospy.Timer(rospy.Duration(5),cb_load,oneshot=True)
rospy.Timer(rospy.Duration(1),cb_param,oneshot=True)
try:
  rospy.spin()
except KeyboardInterrupt:
  print("Shutting down")
