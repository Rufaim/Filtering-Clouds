import numpy as np
import pandas as pd

PATH_DATA = "./train_data/data.csv"
PATH_SEQUENTIAL = "./train_data/data_seq_test_10.npz"

SEQ_LEN = 10

object_class_converter = {"BULLSHIT":2,"OTHER":1,"DRONE":0}

feature_columns = ["speed_stability",
                    "estimated_coverage",
                    "size_mean_orthogonal_gradient",
                    "estimated_speed",
                    "estimated_Mahalanobis_distance",
                    "uavity",
                    "speed_stability_ratio",
                    "speed_direction_stability",
                    "speed_atan2",
                    "acceleration_atan2",
                    "mass_centre_x",
                    "mass_centre_y",
                    "bbox_width",
                    "bbox_height",
                    "speed_stability_std",
                    "estimated_coverage_std",
                    "size_mean_orthogonal_gradient_std",
                    "estimated_speed_std",
                    "estimated_Mahalanobis_distance_std",
                    "uavity_std",
                    "speed_stability_ratio_std",
                    "speed_direction_stability_std",
                    "speed_atan2_std",
                    "acceleration_atan2_std",
                    "mass_centre_x_std",
                    "mass_centre_y_std",
                    "bbox_width_std",
                    "bbox_height_std"
                        ]

data = pd.read_csv(PATH_DATA,index_col=0)
#data = data[data.is_zoom_request>0.0]

sequences = [] #BxTxN
labels = []	#Bx1
ZR = [] #Bx1
Video_id = [] #Bx1
for name,group in data.groupby(["video_id", "object_id"]):
	lab = group.object_class.unique()
	vid = group.video_id.unique()
	if len(lab) > 1 or len(vid) > 1 or\
		group.shape[0]<SEQ_LEN or \
		group.frame_id.diff().sum() != group.shape[0]-1:
		continue
	print("Process #", name, " ", lab)
	for i in range(0,group.shape[0]-SEQ_LEN+1):
		sequences.append([])
		labels.append(lab[0])
		Video_id.append(vid[0])
		zr_ = []
		for j in range(SEQ_LEN):
			sequences[-1].append(list(group.iloc[i+j][feature_columns]))
			zr_.append(group.iloc[i+j].is_zoom_request)
		ZR.append(np.any(zr_).astype(np.int32))

sequences = np.asanyarray(sequences)
labels = np.asanyarray(labels)
np.savez_compressed(PATH_SEQUENTIAL, X=sequences,Y=labels,ZR=ZR,VID=Video_id)	
