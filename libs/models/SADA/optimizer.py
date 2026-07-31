import torch
import torch.optim as optim
from . import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from libs.utils.lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR

def make_optimizer(model, optimizer_config, lr, exclude_params = [], include_params = []):
    """create optimizer based on the parameters to consider that are passed as an argument
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and ('norm' in fpn or 'ln' in fpn or 'bn' in fpn or 'gn' in fpn):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('weight'):
                # weights of all other modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            else:
                # The rest of the parameters are non-decayed
                no_decay.add(fpn)


    # Update the decay and no_decay sets with the parameters to include/exclude
    # Exclude: Exclude all the elements in decay and not decay if they contain in their name one of the elements in exclude_params
    # Include: Include all the elements in decay and not decay if they contain in their name one of the elements in include_params
    
    # NOTE: Up to here, the decay and no_decay contain all the parameters classifier
    
    # Iterate over the decay ones
    to_remove = set()
    for p in decay:
        # Exclude the specified parameters (if any)
        if any(ep in p for ep in exclude_params):
            to_remove.add(p)
            
    if len(include_params) > 0:
        # Discard all the parameters that do not contain any of the elements in include_params
        for p in decay:
            if not any(ip in p for ip in include_params):
                to_remove.add(p)
    decay -= to_remove
    
    # Iterate over the no_decay ones
    to_remove = set()
    for p in no_decay:
        # Exclude the specified parameters (if any)
        if any(ep in p for ep in exclude_params):
            to_remove.add(p)
            
    if len(include_params) > 0:
        # Discard all the parameters that do not contain any of the elements in include_params
        for p in no_decay:
            if not any(ip in p for ip in include_params):
                to_remove.add(p)
    no_decay -= to_remove
    
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=lr,
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=lr,
        )
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(
            optim_groups,
            lr=lr
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config.get("warmup_epochs", 0)
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config.get("warmup_epochs", 0)
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # get eta min
        eta_min = float(optimizer_config["eta_min"])

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get eta min
        eta_min = float(optimizer_config["eta_min"])

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


# -----------------------------------------------------------
# Implementation taken from: https://github.com/val-iisc/SDAT/blob/main/common/utils/sam.py#L5
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
        
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
