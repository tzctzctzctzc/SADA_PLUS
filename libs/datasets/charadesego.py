
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from .datasets import register_dataset
import torch
from .data_utils import truncate_feats

@register_dataset("charadesego")
class CharadesEgo(Dataset):
    def __init__(self, 
                 is_training, 
                 split,
                 anno_path, 
                 feat_path, 
                 mapping_path,
                 num_classes,
                 input_dim,
                 feat_stride,
                 num_frames,
                 default_fps,
                 trunc_thresh,
                 crop_ratio,
                 max_seq_len,
                 crop_alignment=1):
        self.fps = 24
        self.is_training = is_training
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_seq_len = max_seq_len
        self.crop_alignment = int(crop_alignment)
        if self.crop_alignment <= 0:
            raise ValueError("crop_alignment must be a positive integer")
        self.deltaneg = 10 * self.fps
        mapping_path = os.path.join(mapping_path, 'Charades_v1_mapping.txt')
        
        # Define the annotation path according to the split
        if split[0] == "third_training":
            anno_path_split = os.path.join(anno_path, 'CharadesEgo_v1_train.csv')
            self.video_type = 'third'
        elif split[0] == "third_validation":
            anno_path_split = os.path.join(anno_path, 'CharadesEgo_v1_test.csv')
            self.video_type = 'third'
        elif split[0] == "ego_training":
            anno_path_split = os.path.join(anno_path, 'CharadesEgo_v1_train.csv')
            self.video_type = 'ego'
        elif split[0] == "ego_validation":
            anno_path_split = os.path.join(anno_path, 'CharadesEgo_v1_test.csv')
            self.video_type = 'ego'
        
        self.mapping = self.prepare_mapping(mapping_path)
        self.annotations = self.load_annotations(anno_path_split, self.video_type)
        self.data = self.prepare_dataset(feat_path)

        self.db_attributes = {
            'dataset_name': 'charadesego',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }

    
    def get_attributes(self):
        return self.db_attributes

    @classmethod
    def get(cls, args):
        return super(CharadesEgo, cls).get(args, scale=(0.8, 1.0))

    def prepare_mapping(self, mapping_file):
        mapping = pd.read_csv(mapping_file, sep=' ', header=None)
        mapping_dict = {}

        for index, row in mapping.iterrows():
            class_name = row[0]
            
            # Parse the row[2] from the format vxxx to its corresponding int
            verb_class = int(row[2][1:])
            mapping_dict[class_name] = verb_class

        return mapping_dict

    def load_annotations(self, csv_file, video_type='ego'):
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)
        
        # Initialize lists to store the required information
        ids = []
        actions_vids = []
        lengths = []
        charades_videos = []
        
        # Iterate over each row of the dataframe
        for index, row in df.iterrows():
            # Extract the required information from the row
            vid_id = row['id']
            if video_type == 'ego':
                if 'EGO' in vid_id:
                    ids.append(vid_id)
                    actions_vids.append(row['actions'])
                    lengths.append(row['length'])
                    charades_videos.append(row['charades_video'])
            else:
                if 'EGO' not in vid_id:
                    ids.append(vid_id)
                    actions_vids.append(row['actions'])
                    lengths.append(row['length'])
                    charades_videos.append(row['charades_video'])
        
        
        segs_final, act_labels_final = [], []
        
        # Process the actions and obtain the labels (start, end, class)
        for actions_vid in actions_vids:
            try:
                actions_vids = actions_vid.split(';')
            except:
                #print("Problem with the annotations")
                pass
            
            segs_vid, act_labels = [], []   # Actions of a given video
            for action in actions_vids:
                # This contains class id, start, end (in seconds)
                action_info = action.split()
                if len(action_info) != 3:
                    pass
                class_id, start, end = action_info
                
                # Convert the seconds into temporal features (just like EK)
                fps = 24
                num_frames_per_feat = 32
                feat_stride = 8 # overlap between features
                start = (float(start) * fps - 0.5 * num_frames_per_feat) / feat_stride
                end = (float(end) * fps - 0.5 * num_frames_per_feat) / feat_stride
                
                # Map the class label to the verb class to reduce the difficulty of the task using the mapping file
                verb_id = self.mapping[class_id]
                
                # Optional: Used to discard all the verbs that are not in our chosen list
                if verb_id not in [23, 16, 18, 29, 8, 22, 12, 4, 19, 9]:
                    continue
                else:
                    verb_id = [23, 16, 18, 29, 8, 22, 12, 4, 19, 9].index(verb_id)
                    

                seg = [start, end]
                action_label = verb_id
                segs_vid.append(seg)
                act_labels.append(action_label)

            segs_final.append(segs_vid)
            act_labels_final.append(act_labels)
            
        # Return the extracted information
        return ids, segs_final, act_labels_final, lengths, charades_videos

    
    def prepare_dataset(self, data_path):
        datadir = data_path
        data_vids = []
        meta = []
        segments = []
        act_labels_vids = []

        n_vids = len(self.annotations[0])
        for i in range(n_vids):
            vid = self.annotations[0][i]
            segs_vid = self.annotations[1][i]
            act_labels = self.annotations[2][i]
            lengths = self.annotations[3][i]
            charades_videos = self.annotations[4][i]
            
            iddir = f"{datadir}/{vid}"
            
            # Load the data
            data = torch.from_numpy(np.load(iddir + '.npy'))
            
            # Convert from TxC to CxT dimensionality
            try:
                data = data.transpose(1, 0)
            except:
                print("Problem with the data")
                pass
            
            segments.append(segs_vid) # shape: Nx2
            act_labels_vids.append(act_labels)   # shape: N

            meta.append({'video_id': vid,
                        'fps': 24,
                        'duration': lengths,
                        'feat_stride': 8,
                        'video_length': data.shape[1],
                        'feat_num_frames': 32,
                        'use_temporal_grid_feats': True
            })
            
            data_vids.append(data)
            
        return {"data": data_vids,  
                "segments": segments,
                "act_labels": act_labels_vids,
                "meta": meta}
        
        
    def __getitem__(self, index):
        data = self.data['data'][index]
        segments = torch.tensor(self.data['segments'][index])
        labels = torch.tensor(self.data['act_labels'][index])
        meta = self.data['meta'][index]
        
        
        # return a data dict
        data_dict = {'feats'           : data,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'metadata'        : meta,   # dict
                     }

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict,
                self.max_seq_len,
                self.trunc_thresh,
                self.crop_ratio,
                crop_alignment=self.crop_alignment,
            )
        
        return data_dict
          
    def __len__(self):
        return len(self.data['data'])
