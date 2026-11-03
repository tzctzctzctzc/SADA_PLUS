from tools.background_holdout_curve import split_videos, EPOCHS, RATES


def test_video_holdout_is_order_independent_disjoint_complete():
    videos=[f'video{i}' for i in range(51)]
    a=split_videos(videos);b=split_videos(videos[::-1])
    assert a==b and len(a['heldout'])==11 and len(a['fit'])==40
    assert not set(a['fit'])&set(a['heldout'])
    assert set(a['fit']+a['heldout'])==set(videos)


def test_fixed_protocol_does_not_select_epoch_or_rate():
    assert EPOCHS==(1,5,10,20) and RATES==(.005,.01,.02,.05)
    a=split_videos(['a','b']);assert len(a['fit'])==len(a['heldout'])==1
