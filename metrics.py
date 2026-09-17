import numpy as np
from skimage.morphology import skeletonize


class SegMetrics:
    """语义分割指标累计器：mIoU / Accuracy / Precision / Recall / F1 / Dice / clDice。

    update() 逐样本累计混淆矩阵与骨架交并，summary() 输出汇总指标。
    """

    def __init__(self, num_classes, crack_class=1):
        self.num_classes = num_classes
        self.crack_class = crack_class
        n = num_classes
        self.inter = np.zeros(n, dtype=np.float64)
        self.union = np.zeros(n, dtype=np.float64)
        self.tp = np.zeros(n, dtype=np.float64)
        self.fp = np.zeros(n, dtype=np.float64)
        self.fn = np.zeros(n, dtype=np.float64)
        self.correct = 0.0
        self.total = 0.0
        # clDice 累计量
        self.s_pred_tp = 0.0   # |S_pred ∩ V_label|
        self.s_lab_tp = 0.0    # |S_label ∩ V_pred|
        self.s_pred_sum = 0.0  # |S_pred|
        self.s_lab_sum = 0.0   # |S_label|

    def update(self, pred, label):
        pred = np.asarray(pred).astype(np.int64)
        label = np.asarray(label).astype(np.int64)
        for c in range(self.num_classes):
            pc = pred == c
            lc = label == c
            self.inter[c] += (pc & lc).sum()
            self.union[c] += (pc | lc).sum()
            self.tp[c] += (pc & lc).sum()
            self.fp[c] += (pc & ~lc).sum()
            self.fn[c] += (~pc & lc).sum()
        self.correct += (pred == label).sum()
        self.total += label.size

        c = self.crack_class
        if c < self.num_classes:
            pb = (pred == c).astype(np.uint8)
            lb = (label == c).astype(np.uint8)
            sp = skeletonize(pb)
            sl = skeletonize(lb)
            self.s_pred_tp += (sp & lb).sum()
            self.s_lab_tp += (sl & pb).sum()
            self.s_pred_sum += sp.sum()
            self.s_lab_sum += sl.sum()

    def summary(self):
        iou = self.inter / (self.union + 1e-10)
        prec = self.tp / (self.tp + self.fp + 1e-10)
        rec = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        res = {
            "miou": float(iou.mean()),
            "accuracy": float(self.correct / max(self.total, 1.0)),
            "precision": float(prec.mean()),
            "recall": float(rec.mean()),
            "f1": float(f1.mean()),
            "iou_per_class": [round(float(x), 6) for x in iou],
            "precision_per_class": [round(float(x), 6) for x in prec],
            "recall_per_class": [round(float(x), 6) for x in rec],
            "f1_per_class": [round(float(x), 6) for x in f1],
        }

        c = self.crack_class
        if c < self.num_classes:
            res["dice_crack"] = float(f1[c])
            res["iou_crack"] = float(iou[c])
            tprec = self.s_pred_tp / (self.s_pred_sum + 1e-10)
            tsens = self.s_lab_tp / (self.s_lab_sum + 1e-10)
            res["cldice_crack"] = float(2 * tprec * tsens / (tprec + tsens + 1e-10))
        return res


def cldice(pred_bin, label_bin):
    """单图 clDice（二值输入 0/1）。"""
    pred_bin = np.asarray(pred_bin).astype(np.uint8)
    label_bin = np.asarray(label_bin).astype(np.uint8)
    sp = skeletonize(pred_bin)
    sl = skeletonize(label_bin)
    if sp.sum() == 0 and sl.sum() == 0:
        return 1.0
    tprec = (sp & label_bin).sum() / (sp.sum() + 1e-10)
    tsens = (sl & pred_bin).sum() / (sl.sum() + 1e-10)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens + 1e-10))
