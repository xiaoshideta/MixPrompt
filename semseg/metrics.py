import torch
from torch import Tensor
from typing import Tuple

import numpy as np


class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc*100, acc_cls*100, mean_iou*100, fw_iou*100
    # (
    #         {
    #             "pixel_acc: ": acc,
    #             "class_acc: ": acc_cls,
    #             "mIou: ": mean_iou,
    #             "fwIou: ": fw_iou,
    #         },
    #         cls_iu,
    #     )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious[ious.isnan()]=0.
        
        # miou = ious.mean().item()
        
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1[f1.isnan()]=0.
        # mf1 = f1.mean().item()
        
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc[acc.isnan()]=0.
        # macc = acc.mean().item()
        
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

