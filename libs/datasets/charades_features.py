"""统一的Charades I3D接口：保留发布成员，支持无动作视频和无标注目标流。"""
import numpy as np
import torch

from .epic_kitchens import EpicKitchensDataset
from .data_utils import truncate_feats


class CharadesFeatures(EpicKitchensDataset):
    def __init__(self,is_training,split,**options):
        # 复用已核验的秒→特征坐标接口；裁剪在本类执行，空动作不进行GT重试。
        super().__init__(False,split,**options)
        self.is_training=is_training
        self.db_attributes['dataset_name']='charades-ego-third-to-first'

    def _load_json_db(self,path):
        records,labels=super()._load_json_db(path)
        for row in records:
            if row['segments'] is None:
                row['segments']=np.empty((0,2),dtype=np.float32)
                row['labels']=np.empty((0,),dtype=np.int64)
        return records,labels

    def __getitem__(self,index):
        row=self.data_list[index]
        with np.load(self.feat_folder+'/'+self.file_prefix+row['id']+self.file_ext) as data:
            array=data['feats'][::self.downsample_rate]
        feats=torch.from_numpy(np.ascontiguousarray(array.T))
        stride=self.feat_stride*self.downsample_rate
        segments=torch.from_numpy((row['segments']*row['fps']-.5*self.num_frames)/stride)
        labels=torch.from_numpy(row['labels'])
        sample=dict(feats=feats,segments=segments,labels=labels,metadata=dict(
            video_id=row['id'],fps=row['fps'],duration=row['duration'],feat_stride=stride,
            feat_num_frames=self.num_frames,video_length=feats.shape[1],use_temporal_grid_feats=True))
        if self.is_training:
            sample=truncate_feats(sample,self.max_seq_len,self.trunc_thresh,self.crop_ratio,
                                  has_action=bool(labels.numel()) and not self.unlabeled)
        return sample
