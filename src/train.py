# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from IEF.config import cfg
#import render_data_layer.imdata as imdata
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, my_imkptdb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

         
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
        print ('Just copied')
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_imdata_list(my_imkptdb._imdata_list)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        #if cfg.TRAIN.BBOX_REG:
            ## save original values
            #orig_0 = net.params['bbox_pred'][0].data.copy()
            #orig_1 = net.params['bbox_pred'][1].data.copy()

            ## scale and shift with bbox reg unnormalization; then save snapshot
            #net.params['bbox_pred'][0].data[...] = \
                    #(net.params['bbox_pred'][0].data *
                     #self.bbox_stds[:, np.newaxis])
            #net.params['bbox_pred'][1].data[...] = \
                    #(net.params['bbox_pred'][1].data *
                     #self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        #if cfg.TRAIN.BBOX_REG:
            ## restore net to original state
            #net.params['bbox_pred'][0].data[...] = orig_0
            #net.params['bbox_pred'][1].data[...] = orig_1

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

#def get_training_imdata(imdb):
    ##TO DO
    #if cfg.TRAIN.USE_FLIPPED:
        #print 'Appending horizontally-flipped training examples...'
        #imdb.append_flipped_images()
        #print 'done'
##FIX
    #return imdb.imdata[istrain==1]

def train_net(solver_prototxt, my_imkptdb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, my_imkptdb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
