
"""The data layer used during training to train IEF network.
PoseDataLayer implements a Caffe Python layer for training IEF 
models for human pose estimation.
"""

import caffe
from IEF.config import cfg
from render_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class PoseDataLayer(caffe.Layer):
    """IEF layer used for training Human Pose Estimation"""
	def setup(self, bottom, top):
		# parse the layer parameter string, which must be valid YAML
		layer_params = yaml.load(self.param_str_)
		self._num_keypoints  = layer_params['num_keypoints']
		self._nDim           = layer_params['nDim']
		self._gaussian_width = layer_params['gaussian_width']
		#self._mean_pose_file = layer_params['mean_pose_file']
		print("Parameters:: _num_keypoints:",self._num_keypoints,
		 "_nDim:",self._nDim)
		self._name_to_top_map = {
				'data': 0,
				'targets': 1}
		# data blob: holds a batch of N images, each with 3 channels
		# The height and width (100 x 100) are dummy values
		top[0].reshape(1, 3+self._num_keypoints, cfg.CROP_SIZE, cfg.CROP_SIZE)
		top[1].reshape(1, self._num_keypoints*self._nDim,1,1)


    def _shuffle_imdata_list_inds(self):
        """Randomly permute the training imdata_list."""
        self._perm = np.random.permutation(np.arange(len(self._imdata_list)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the imdata_list indices for the next minibatch."""
        if self._cur + cfg.TRAIN.BATCH_SIZE >= len(self._imdata_list):
            self._shuffle_imdata_list_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._imdata_list[i] for i in db_inds]
            return get_minibatch(minibatch_db,self._gaussian_width)

    def set_imdata_list(self, imdata_list):
        """Set the imdata_list to be used by this layer during training."""
        self._imdata_list = imdata_list
        self._shuffle_imdata_list_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._imdata_list,
                                                 self._num_keypoints,
                                                 self._nDim,
                                                 self._gaussian_width)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

     

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, imdata_list, num_keypoints, nDim, gaussian_width):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._imdata_list = imdata_list
        self._gaussian_width=_gaussian_width
        self._num_keypoints = num_keypoints
        self._nDim = nDim
        self._perm = None
        self._cur = 0
        self._shuffle_imdata_list_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_imdata_list_inds(self):
        """Randomly permute the training imdata_list."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._imdata_list)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the imdata_list indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.BATCH_SIZE >= len(self._imdata_list):
            self._shuffle_imdata_list_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return db_inds
      
    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._imdata_list[i] for i in db_inds]
            return get_minibatch(minibatch_db,self._gaussian_width)
    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._imdata_list[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db,self._gaussian_width)
            self._queue.put(blobs)
