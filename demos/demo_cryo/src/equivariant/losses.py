import torch
import torch.nn as nn
from deepinv.loss import Loss
from .transform import Rotate3D

class ObsLoss(Loss):
    def __init__(self, physics, weight: float = 1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self._physics = physics
        self._criteria = nn.MSELoss(reduction="mean")

    def forward(
        self,
        x: torch.Tensor,        # EVN patch (y1)
        y: torch.Tensor,        # ODD patch (y2)
        x_net: torch.Tensor,    # f(y) = f(EVN)
        physics,
        model,
        **kwargs,
    ) -> torch.Tensor:
        est_evn = x_net                          # f(x)
        est_odd = kwargs.get("y_net")            # f(y), pre-computed by trainer
        if est_odd is None:
            est_odd = model(y)

        # ob_loss: y2 - A(f(y1)) and y1 - A(f(y2))
        loss = self._criteria(y, self._physics.A(est_evn)) + \
               self._criteria(x, self._physics.A(est_odd))
        return self.weight * loss


class EqLoss(Loss):
    def __init__(
        self,
        physics,
        transform: Rotate3D,
        weight: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.weight = weight
        self._physics = physics
        self._transform = transform
        self._criteria = nn.MSELoss(reduction="mean")
        self._valid_k_sets = list(range(len(Rotate3D._KSET)))

    def _rotate_batch_per_sample(self, x: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(x.shape[0]):
            out.append(self._transform.transform(x[i:i + 1], k_idx=int(k_indices[i].item())))
        return torch.cat(out, dim=0)

    def forward(
        self,
        x: torch.Tensor,        # EVN patch (y1)
        y: torch.Tensor,        # ODD patch (y2)
        x_net: torch.Tensor,    # f(EVN)
        physics,
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        est_evn = x_net                          # f(EVN)
        est_odd = kwargs.get("y_net")            # f(ODD)
        if est_odd is None:
            est_odd = model(y)

        pool = self._valid_k_sets
        bsz = x.shape[0]
        rand_idx = torch.randint(len(pool), (bsz,), device=x.device)
        k_indices = torch.tensor([pool[int(i.item())] for i in rand_idx], device=x.device)

        z1 = self._rotate_batch_per_sample(est_evn, k_indices)
        z2 = self._rotate_batch_per_sample(est_odd, k_indices)
        
        # loss = z - f(A(z))
        loss1 = self._criteria(z1, model(self._physics.A(z2)))
        loss2 = self._criteria(z2, model(self._physics.A(z1)))
        
        return self.weight * (loss1 + loss2)
