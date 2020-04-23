import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from utils.box_utils import match_source_target_prediction_for_rois

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(DEVICE)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(DEVICE)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class DomainAdapt(nn.Module):
    def __init__(self, method="CORAL"):
        super(DomainAdapt, self).__init__()

        methods = {
            "CORAL": self.coral,
            "cross_entropy": self.cross_entropy,
            "wasserstein": self.wasserstein
        }
        self.method = methods[method]
        if method == "wasserstein":
            self.wasserstein = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean').to(DEVICE)

    def forward(self, source, target):
        # return self.method(source, target)
        return self.cross_entropy(source, target)

    def coral(self, source, target):
        d = source.size(1)
        ns, nt = source.size(0), target.size(0)
        # source covariance
        tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
        cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / ns

        # target covariance
        tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
        ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / nt
        # frobenius norm
        loss = (cs - ct).pow(2).sum().sqrt()
        loss = loss / (4 * d * d)
        return loss

    def cross_entropy(self, source, target):
        source = torch.log_softmax(source, dim=-1).expand_as(target)
        loss = source * torch.log_softmax(target, dim=-1)
        print(loss)
        return loss

    def chisquare(self, source, target):
        source = torch.softmax(source, dim=1).expand_as(target)
        target = torch.softmax(target, dim=1)
        loss = torch.sum(torch.pow(target - source, 2) / source)
        return loss

    def wasserstein(self, source, target):
        source.unsqueeze_(-1)
        target.unsqueeze_(-1)
        dist, P, C = self.wasserstein(target, source)
        return dist


class SoftTarget(nn.Module):
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=-1), F.softmax(out_t/self.T, dim=-1), reduction='batchmean') * self.T * self.T
        return loss


class AdaptLoss(nn.Module):
    def __init__(self, num_classes=21, negpos_ratio=3, iou_thresh=0.3, T=2.0):
        super(AdaptLoss, self).__init__()
        self.num_classes = num_classes
        self.negpos_ratio = negpos_ratio
        self.iou_thresh = iou_thresh
        self.T = T

    def forward(self, rois, prediction, source_rois, source_conf):
        """

        :param rois: scaled tensor [batchsize, 1, top_k, 4]
        :param prediction: tensor [batchsize, top_k, num_classes]
        :param source_rois
        :param source_conf: [batchsize, top_k, num_classes]
        :return:
        """
        batchsize = rois.size(0)
        num_rois = prediction.size(1)
        assign_labels = torch.zeros(size=(batchsize, num_rois, prediction.size(-1))).long().to(config.device)

        # 给每一个roi按照与true的iou最大分配标签，如果iou小于阈值则让其为0背景
        for idx in range(batchsize):
            match_source_target_prediction_for_rois(rois[idx][0], source_rois[idx][0], source_conf[idx], assign_labels, idx, self.iou_thresh)

        assign_labels = assign_labels.view(-1, self.num_classes).float() / self.T
        prediction = prediction.view(-1, self.num_classes).float() / self.T
        pos = torch.sum(prediction, dim=1) > 0

        source_pred = torch.log_softmax(assign_labels, dim=-1).expand_as(prediction)
        loss = source_pred * torch.log_softmax(prediction, dim=-1)
        loss = loss[pos].mean()
        return loss

if __name__ == '__main__':
    # rois = torch.rand(size=(1,1,5,4))
    # source_rois = torch.rand(size=(1, 1, 3, 4))
    # rois[:,:,:,2:] = rois[:,:,:,:2] + 0.5
    # source_rois[:,:,:,2:] = source_rois[:,:,:,:2] + 0.5
    #
    # predict = torch.randn(size=(1, 5, 3))
    # source_conf = torch.randn(size=(1, 3, 3))
    # domain = AdaptLoss(3)
    # domain(rois, predict, source_rois, source_conf)
    source = torch.tensor([[0.0, 1.0, 1.0]])
    target = torch.tensor([[0.1, 0.7, 0.8]])
    domain = DomainAdapt(method="cross_entropy")
    loss = domain(source, target)
    print(loss)

