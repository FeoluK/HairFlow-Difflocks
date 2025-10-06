from callbacks.callback import *
import numpy as np
from gloss  import *

class ViewerCallback(Callback):

    def __init__(self, viewer_config_path, experiment_name):
        gloss_setup_logger(log_level=LogLevel.Info) 
        self.viewer=Viewer(viewer_config_path)
        self.scene=self.viewer.get_scene()

        self.first_time=True
        
        self.visualize_every_x_iters=100
        self.visualize_strand_at_idx=None
        # self.visualize_strand_at_idx=0

    def after_forward_pass(self, phase, gt_cloud=None, pred_cloud=None, **kwargs):

        # if phase.iter_nr==1000:
            # exit(1)

        if phase.iter_nr%self.visualize_every_x_iters==0:
            if self.visualize_strand_at_idx is not None:
                gt_cloud=gt_cloud[self.visualize_strand_at_idx,:,:]
            gt_ent = self.scene.get_or_spawn_renderable("gt_ent")
            gt_ent.insert(Verts(gt_cloud.reshape(-1,3).cpu().numpy()))
            gt_ent.insert(Colors(gt_cloud.reshape(-1,3).cpu().numpy())) #just to avoid verts being different than Colors when we switch nr_verts for the same entity
            if self.first_time:
                gt_ent.insert(VisPoints(show_points=True, \
                                    point_size=3.0, \
                                    color_type=PointColorType.Solid))
                

            #pred
            if self.visualize_strand_at_idx is not None:
                pred_cloud=pred_cloud[self.visualize_strand_at_idx,:,:]
            pred_ent = self.scene.get_or_spawn_renderable("pred_ent")
            pred_ent.insert(Verts(pred_cloud.reshape(-1,3).cpu().numpy()))
            pred_ent.insert(Colors(pred_cloud.reshape(-1,3).cpu().numpy())) #just to avoid verts being different than Colors when we switch nr_verts for the same entity
            if self.first_time:
                pred_ent.insert(VisPoints(show_points=True, \
                                    point_size=3.0, \
                                    point_color=[0.1, 0.2, 0.8, 1.0],
                                    color_type=PointColorType.Solid))
        
        #render
        self.viewer.start_frame()
        self.viewer.update() 


        self.first_time=False





# viewer=Viewer()
# scene=viewer.get_scene()

# mesh = scene.get_or_spawn_renderable("test")
# mesh.insert(Verts(
#     np.array([ 
#     [0,0,0],
#     [0,1,0],
#     [1,0,0],
#     [1.5,1.5,-1]
#     ], dtype = "float32") ))
# mesh.insert(Colors(
#     np.array([ 
#     [1,0,0],
#     [0,0,1],
#     [1,1,0],
#     [1,0,1]
#     ], dtype = "float32")))

# mesh.insert(VisPoints(show_points=True, \
#                       show_points_indices=True, \
#                       point_size=10.0, \
#                       color_type=PointColorType.PerVert))

# while True:
#     viewer.start_frame()
#     viewer.update() 


    