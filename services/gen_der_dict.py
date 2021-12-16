#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:14:52 2020

@author: prachi
"""

import pickle
import numpy as np

der='swbd_diar/exp_new/callhome/plda_oracle/der.scp'
der_pickle = 'swbd_diar/exp_new/callhome/plda_oracle/derdict'
der=open(der,'r').readlines()
DER={}
for line in der[2:-1]:
    fname = line.split()[0]
    val = float(line.split()[1])
    DER[fname] = val

pickleobj=open(der_pickle,'wb')
pickle.dump(DER,pickleobj)
pickleobj.close()
