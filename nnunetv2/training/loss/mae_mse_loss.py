from torch import nn
import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch.nn import functional as F

class MAEMSELoss(nn.Module):
    def __init__(self, ddp: bool = False):
        super().__init__()
        self.ddp = ddp
        # self.loss = nn.MSELoss(reduction="none")

    def forward(
            self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
            # mask = AllGatherGrad.apply(mask).sum(0)
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * (1 - mask)) / torch.sum((1 - mask))

        return reconstruction_loss

class MAEL1SSIMLoss(nn.Module):
    def __init__(
        self,
        loss_weights: tuple[float, float] = (1.0, 0.5),  # (l1, ssim)
        per_block_norm: bool = True,
        mask_block_size: tuple[int, int, int] = (16, 16, 16),
        ssim_window: int = 7,
        ddp: bool = False,
    ):
        super().__init__()
        self.l1_weight, self.ssim_weight = loss_weights
        self.per_block_norm = per_block_norm
        self.mask_block_size = mask_block_size
        self.ssim_window = ssim_window
        self.ddp = ddp
        self.l1 = nn.L1Loss(reduction="none")
        self.eps = 1e-6

    def _ssim_3d(self, pred, target):
        # pred, target: [B, C, D, H, W]
        ws, pad = self.ssim_window, self.ssim_window // 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ch = pred.shape[1]
        kernel = torch.ones(ch, 1, ws, ws, ws, device=pred.device, dtype=pred.dtype)
        kernel = kernel / (ws ** 3)
        filt = lambda z: F.conv3d(z, kernel, padding=pad, groups=ch)

        mu_p, mu_t = filt(pred), filt(target)
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
        sig_p2 = filt(pred*pred) - mu_p2
        sig_t2 = filt(target*target) - mu_t2
        sig_pt = filt(pred*target) - mu_pt
        ssim_map = (((2*mu_pt + C1)*(2*sig_pt + C2)) /
                    ((mu_p2 + mu_t2 + C1)*(sig_p2 + sig_t2 + C2)))
        return ssim_map  # [B, C, D, H, W], keep per-voxel for masking

    def forward(self, model_output, target, mask):
        # all: [B, C, D, H, W];  mask: 1 = visible, 0 = masked
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
            mask = AllGatherGrad.apply(mask).sum(0)

        score = 1.0 - mask                      # 1 at MASKED voxels (what we score)

        if self.per_block_norm:
            target = self._normalize_per_block(target)

        # ---- L1, masked-only ----
        diff = self.l1(model_output, target)    # [B,C,D,H,W]
        l1 = (diff * score).sum() / (score.sum() + self.eps)

        # ---- SSIM, masked-only ----
        ssim_map = self._ssim_3d(model_output, target)   # [B,C,D,H,W]
        ssim_masked = (ssim_map * score).sum() / (score.sum() + self.eps)
        ssim_term = 1.0 - ssim_masked

        return self.l1_weight * l1 + self.ssim_weight * ssim_term

    def _normalize_per_block(self, x):
        B, C, D, H, W = x.shape
        bd, bh, bw = self.mask_block_size
        assert D % bd == 0 and H % bh == 0 and W % bw == 0, \
            f"patch {(D,H,W)} not divisible by block {self.mask_block_size}; " \
            f"per-block norm would mismatch the mask's border handling"
        # normalize each mask_block by its own mean/std (norm_pix_loss, dense version)
        
        # reshape into blocks
        x = x.reshape(B, C, D//bd, bd, H//bh, bh, W//bw, bw)
        x = x.permute(0,1,2,4,6,3,5,7).contiguous()       # [B,C, nD,nH,nW, bd,bh,bw]
        shp = x.shape
        x = x.reshape(*shp[:5], -1)                        # [B,C,nD,nH,nW, block]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        x = x.reshape(shp).permute(0,1,2,5,3,6,4,7).contiguous().reshape(B,C,D,H,W)
        return x
    
    
class MAEL1LossMSELoss(nn.Module):
    def __init__(self, loss_weights: tuple[float, float] = (1, 1), add_normal_loss: bool = False, ddp: bool = False):
        super().__init__()
        self.loss_weights = loss_weights
        self.mse = nn.MSELoss(reduction="mean")
        self.l1 = nn.L1Loss(reduction="mean")
        self.add_normal_loss = add_normal_loss
        self.ddp = ddp

    def forward(
            self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
            # mask = AllGatherGrad.apply(mask).sum(0)

        loss = 0
        if self.add_normal_loss:
            loss = self.l1(model_output, target) * self.loss_weights[0] + self.mse(model_output, target) * \
                   self.loss_weights[1]
            loss /= sum(self.loss_weights)

        model_output = model_output * (1 - mask)
        target = target * (1 - mask)
        masked_loss = (self.l1(model_output, target) * self.loss_weights[0] + self.mse(model_output, target) * \
                 self.loss_weights[1])

        masked_loss /= sum(self.loss_weights)
        masked_loss /= (torch.sum((1 - mask)) / torch.sum(torch.ones_like(mask)))

        loss += masked_loss
        return loss


class MAEL1Loss(nn.Module):
    def __init__(self, add_normal_loss: bool = False, ddp: bool = False):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")
        self.add_normal_loss = add_normal_loss
        self.ddp = ddp

    def forward(
            self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
            # mask = AllGatherGrad.apply(mask).sum(0)
        loss = 0
        if self.add_normal_loss:
            loss = (loss + self.l1(model_output, target))
            
        model_output = model_output * (1 - mask)
        target = target * (1 - mask)
        masked_loss = self.l1(model_output, target)
        masked_loss /= (torch.sum((1 - mask)) / torch.sum(torch.ones_like(mask)))
        
        loss += masked_loss
        
        return loss


class L1LossMSELoss(nn.Module):
    def __init__(self, loss_weights: tuple[float, float] = (1, 1), ddp: bool = False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.loss_weights = loss_weights
        self.ddp = ddp

    def forward(
            self, model_output: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
        loss = self.l1(model_output, target) * self.loss_weights[0] + self.mse(model_output, target) * \
               self.loss_weights[1]

        loss /= sum(self.loss_weights)

        return loss


class L1Loss(nn.Module):
    def __init__(self, ddp: bool = False):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ddp = ddp

    def forward(self, model_output: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.ddp:
            model_output = AllGatherGrad.apply(model_output).sum(0)
            target = AllGatherGrad.apply(target).sum(0)
        loss = self.l1(model_output, target) 

        return loss



if __name__ == '__main__':
    # Check whether MAEMSE is equal in two algorithms

    mse_loss = MAEL1LossMSELoss((0, 1))
    mae_mse_loss = MAEMSELoss()
    arr = torch.randint(0, 10, size=(24, 24)).to(torch.float32)
    arr2 = torch.randint(0, 10, size=(24, 24)).to(torch.float32)
    mask = torch.randint(0, 1, size=(24, 24)).to(torch.float32)
    loss1 = mse_loss(arr, arr2, mask)
    loss2 = mae_mse_loss(arr, arr2, mask)
    print(loss1, loss2)
