# from permuto_sdf  import TrainParams
from callbacks.callback import *
from callbacks.wandb_callback import *
from callbacks.state_callback import *
from callbacks.phase import *


def create_callbacks(with_tensorboard, with_visualizer, experiment_name, viewer_config_path=None):
    cb_list = []
    if(with_tensorboard):
        from callbacks.tensorboard_callback import TensorboardCallback #we put it here in case we don't have tensorboard installed
        tensorboard_callback=TensorboardCallback(experiment_name)
        cb_list.append(tensorboard_callback)
    if(with_visualizer):
        from callbacks.viewer_callback import ViewerCallback #we put it here because we might not have the visualizer package installed
        viewer_callback=ViewerCallback(viewer_config_path, experiment_name)
        cb_list.append(viewer_callback)
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    return cb
