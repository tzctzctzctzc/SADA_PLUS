from tools.vlm_background_inside import inside_row


def test_only_context_removed_and_static_physical_identity():
    row=dict(indices=list(range(16)),times=[i*.1 for i in range(16)],frame_records=[{'sha':str(i)} for i in range(16)],label=4,proposal=[.2,1.4],observable=True)
    new=inside_row(row)
    assert len(row['indices']) == 16 and len(new['indices']) == 12
    assert new['indices'] == list(range(2,14))
    assert new['times'] == row['times'][2:14]
    assert new['frame_records'][6] == row['frame_records'][8]
    assert new['proposal'] == row['proposal'] and new['label'] == row['label']
