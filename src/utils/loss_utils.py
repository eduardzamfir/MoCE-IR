import math

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)



class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, epsilon=1e-6, alpha=0.1):
        """
        Focal L1 Loss with adjusted weighting for output values in [0, 1].
        
        Args:
            gamma (float): Focusing parameter. Larger gamma focuses more on hard examples.
            epsilon (float): Small constant to prevent weights from being zero.
            alpha (float): Scaling factor to normalize error values.
        """
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha  # Scaling factor to prevent error values from being too small.

    def forward(self, pred, target):
        """
        Compute the Focal L1 Loss between the predicted and target images.
        
        Args:
            pred (torch.Tensor): Predicted image [b, c, h, w].
            target (torch.Tensor): Ground truth image [b, c, h, w].
        
        Returns:
            torch.Tensor: Scalar Focal L1 Loss.
        """
        # Compute the absolute error (L1 Loss) and scale it by alpha
        abs_err = torch.abs(pred - target) / self.alpha
        
        # Apply a logarithmic transformation to the error to prevent very small weights
        focal_weight = (torch.log(1 + abs_err + self.epsilon)) ** self.gamma
        
        # Compute the weighted loss
        focal_l1_loss = focal_weight * abs_err
        
        # Return the mean loss across all pixels
        return focal_l1_loss.mean()
    
    
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
    
class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        """
        Scheduler for Gumbel-Softmax temperature that decreases using a cosine annealing schedule.

        Args:
        - start_temp (float): Initial temperature (e.g., 5.0).
        - end_temp (float): Final temperature (e.g., 0.01).
        - total_steps (int): Total number of steps/epochs to anneal over.
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):
        """
        Get the temperature value for the current step, following a cosine annealing schedule.
        
        Args:
        - step (int): Current step or epoch.
        
        Returns:
        - temperature (float): The temperature for the Gumbel-Softmax at this step.
        """
        if step >= self.total_steps:
            return self.end_temp
        
        # Cosine annealing formula to compute the temperature
        cos_inner = math.pi * step / self.total_steps
        #temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + math.cos(cos_inner))
        temp = self.start_temp + 0.5 * (self.end_temp - self.start_temp) * (1 - math.cos(cos_inner))
        
        return temp
