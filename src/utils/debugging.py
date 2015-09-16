#!/usr/bin/env python

# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under The MIT License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

import imkptdb
import pprint
import numpy as np
import sys

def main():
 data_file='/data0/katef/Code/IEF/matlabcode/data_file_catalin.hdf5'
 imname_file='/data0/katef/Code/IEF/matlabcode/imnames_file_catalin.txt'
 catalin_imkptdb=imkptdb.Imkptdb('catalin_2D')
 catalin_imkptdb.load_from_files(imname_file,data_file)
 print("num images:",catalin_imkptdb.num_images)
 return

if __name__ == '__main__':
    main()
