#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import copy
from rovi_utils import tflib

def solve(pcd):
  pcdw=copy.deepcopy(pcd)
  obb=pcdw.get_oriented_bounding_box()
  Tm=np.eye(4,dtype=float)
  Tm[:3,3]=np.array(obb.center[:3]).T
  return Tm

