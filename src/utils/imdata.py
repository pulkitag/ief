## @package imdata
# Defines Classes for storing the data
#

import matplotlib.pyplot as plt

##
# Parent class for data objects
class ImData(object):
	def __init__ (self, imFile=''):
		self.imFile_ = imFile

class Bbox(object):
	def __init__ (self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def plot(self, col='r', th=3.0):
		
			
	

class ImKPtData(ImData):
	def __init__ (self, imFile='', kpts=[], bbox=None):
		'''
			imFile: Path to image file
			
		'''
		pass


