# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

#!/bin/bash
SRC_URL='www.cs.berkeley.edu:~/pulkitag/ief/'
#Download the models
wget -l 0 ${SRC_URL}/models.tar ./
tar -xf models.tar
rm models.tar
#Download the mpii data

