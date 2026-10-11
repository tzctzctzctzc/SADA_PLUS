import pytest
from tools.rq4_relation_quality import original_grid_points, same_instance_pairs, summarize


def test_relationship_means_same_instance_not_same_class():
    anns=[dict(segment=[0,3],label_id=1),dict(segment=[3,6],label_id=1)]
    assert same_instance_pairs([1,2,4],anns)==[True,False]


def test_original_grid_is_not_arbitrary_interpolation():
    points=original_grid_points([2.,8.],dict(fpn_level=1),
        dict(default_fps=30,feat_stride=16,downsample_rate=1,num_frames=32),2)
    assert len(points)==3 and all(2<p<8 for p in points)
    assert all(((p*30-16)/32)==pytest.approx(round((p*30-16)/32)) for p in points)


def test_soft_summary_preserves_low_weight_incorrect_relations():
    rows=[dict(points=[1,2],same_instance_pairs=[True],consistency=.8),
          dict(points=[2,3],same_instance_pairs=[False],consistency=.2)]
    assert summarize(rows,'soft')['weighted_pair_precision']==pytest.approx(.8)
    assert summarize(rows,'hard')['candidates']==1
