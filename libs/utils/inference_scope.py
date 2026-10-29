"""保留双路网络前向，只省去最终不会使用的那一路后处理。"""


def reporting_forward(model, video_list, domain):
    if domain not in ('src', 'tgt'):
        raise ValueError('reporting requires an explicit source/target domain')
    missing = object()
    previous = getattr(model, '_rq3_reporting_domain', missing)
    model._rq3_reporting_domain = domain
    try:
        return model(video_list_src=video_list, video_list_tgt=video_list, mode='test')
    finally:
        if previous is missing:
            del model._rq3_reporting_domain
        else:
            model._rq3_reporting_domain = previous
