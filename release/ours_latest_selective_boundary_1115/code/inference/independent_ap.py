"""复用既有独立NumPy AP匹配器；不依赖官方AP实现。"""
import numpy as np
THRESHOLDS=np.linspace(.1,.5,5)
def overlaps(boxes, truth):
    if not len(truth):
        return np.zeros((len(boxes), 0))
    intersection = np.maximum(0, np.minimum(boxes[:, None, 1], truth[None, :, 1]) - np.maximum(boxes[:, None, 0], truth[None, :, 0]))
    return intersection / np.maximum((boxes[:, 1] - boxes[:, 0])[:, None] + (truth[:, 1] - truth[:, 0])[None, :] - intersection, 1e-08)

class IndependentAP:

    def __init__(self, gt, pred):
        self.scores = pred.score.to_numpy(dtype=float)
        self.n = len(gt)
        self.options = [[] for _ in range(len(pred))]
        g = gt.reset_index(drop=True)
        for (video, ids) in pred.groupby('video-id', sort=False).indices.items():
            gids = np.flatnonzero(g['video-id'].to_numpy() == video)
            matrix = overlaps(pred.iloc[ids][['t-start', 't-end']].to_numpy(dtype=float), g.iloc[gids][['t-start', 't-end']].to_numpy(dtype=float))
            for (pos, row) in zip(ids, matrix):
                order = np.argsort(row)[::-1]
                self.options[pos] = [(int(gids[j]), float(row[j])) for j in order if row[j] >= 0.1]

    def ap(self, keep):
        ids = np.flatnonzero(keep)
        ordered = ids[np.argsort(self.scores[ids])[::-1]]
        if not self.n or not len(ids):
            return np.zeros(5)
        result = []
        for threshold in THRESHOLDS:
            assigned = set()
            hit = np.zeros(len(ids))
            for (rank, pos) in enumerate(ordered):
                for (identity, value) in self.options[pos]:
                    if value < threshold:
                        break
                    if identity not in assigned:
                        assigned.add(identity)
                        hit[rank] = 1
                        break
            precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            result.append(float(envelope[hit.astype(bool)].sum() / self.n))
        return np.asarray(result)
