#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.chdir('/home/asy/Documents/label')
filename = "000001.txt"
file = open(filename)
lines = len(file.readlines())
print ("There are %d lines in %s" %(lines,filename))
