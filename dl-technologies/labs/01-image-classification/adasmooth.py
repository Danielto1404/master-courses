from typing import List, Tuple

import torch
import torch.optim as to


class AdaSmooth(to.Optimizer):
    def __init__(
            self,
            params,
            lr=1.0,
            betas=(0.5, 0.99),
            eps=1e-6,
            weight_decay=0,
            *,
            maximize: bool = False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize
        )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []

            x_t = []
            e_t = []

            lr, betas, eps, weight_decay, maximize = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
                group["maximize"]
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdaSmooth does not support sparse gradients")

                grads.append(p.grad)
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["X_t"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["E_t"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                x_t.append(state["X_t"])
                e_t.append(state["E_t"])

                state["step"] += 1

            ada_smooth_step(
                params_with_grad,
                betas,
                grads,
                x_t=x_t,
                e_t=e_t,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay,
                maximize=maximize
            )

        return loss


def ada_smooth_step(
        params: List[torch.Tensor],
        betas: Tuple[float, float],
        grads: List[torch.T],
        x_t: List[torch.Tensor],
        e_t: List[torch.Tensor],
        *,
        lr: float,
        eps: float,
        weight_decay: float,
        maximize: bool
):
    fast, slow = betas

    for (param, grad, x, e) in zip(params, grads, x_t, e_t):
        grad = grad if not maximize else -grad

        ertaio = param - x
        smooth = (slow - fast) * ertaio + (1 - fast)

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        smooth_sq = smooth ** 2
        gamma = grad ** 2 * smooth_sq
        alpha = (1 - smooth_sq) * e ** 2
        e.add(gamma + alpha)
        std = grad.add(eps).sqrt_()
        delta = grad.div_(std)
        param.add_(delta, alpha=-lr)


__all__ = [
    "AdaSmooth"
]
