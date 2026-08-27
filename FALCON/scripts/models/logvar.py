import torch
import numpy as np

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class LogVarModel(torch.nn.Module):
    r"""
    A LogVar model that takes the losses and outputs the weighted sum.

    Parameters:
        logvar_channels (`int` , *required*, defaults to `128`):
            Channel size for the MLP that estimates the log variance.
    """
    r"""automatically weighted multi-task loss

    Params:
        num_losses: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = LogVarModel(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num_losses=4):
        super(LogVarModel, self).__init__()
        self.params = torch.nn.ParameterList()
        for _ in range(num_losses):
            # params = torch.rand(1, requires_grad=True, dtype=torch.float32)
            # params = torch.ones(1, requires_grad=True, dtype=torch.float32)
            params = torch.zeros(1, requires_grad=True, dtype=torch.float32)
            self.params.append(torch.nn.Parameter(params, requires_grad=True))
        self.params.cuda()
        # self.params = torch.nn.Parameter(torch.randn(1, num_losses))
        self.num_losses = num_losses

    def forward(self, losses):
        r"""
        The [`LogVarModel`] forward method.

        Args:
            The list of Losses (`torch.FloatTensor`): List of losses.

        Returns:
            [`torch.Tensor`]:
                Weighted sum of the losses.
        """
        assert len(losses) == self.num_losses, f"length of losses list: {len(losses)} | num_losses : {self.num_losses}"
        # loss_sum = torch.zeros_like(losses[0])
        loss_sum = 0.0
        for i, loss in enumerate(losses):
            # loss_sum += (0.5 / self.params[i].square()) * loss + torch.log(1 + self.params[i].square())
            # loss_sum += (0.5 / self.params[i].square()) * loss + torch.log(self.params[i])
            # loss_sum += loss * (1 / self.params[i].square()) + torch.log(self.params[i].square())
            # loss_sum += (loss / self.params[i].exp()) + self.params[i]
            assert (loss >= 0).all(), loss[loss<0]
            # if (loss < 0).any():
            #     print('loss is -ve', i, loss.item(), '| weighted loss', (loss.item() / (2 * self.params[i].exp().item())) + self.params[i].item())
            weighted_loss = (0.5 * loss * (-self.params[i]).exp()) + self.params[i]
            # assert (weighted_loss >= 0).all(), f"{weighted_loss[weighted_loss < 0]} | {loss[weighted_loss < 0]} | {self.params[i].item()} | {0.5 * (-self.params[i]).exp().item()}"
            loss_sum = loss_sum + weighted_loss
            # assert (loss_sum >= 0).all(), f"{loss_sum[loss_sum < 0]} | loss_sum org = {(loss_sum >= weighted_loss).all()}"
            # if (loss_sum < 0).any():
            #     print('loss sum has become -ve', (loss / (2 * self.params[i].exp()) + self.params[i]).mean().item(), '| org loss prior addition:', (loss_sum - ((loss / (2 * self.params[i].exp())) + self.params[i])).mean().item(), '| new loss:', loss_sum.mean().item())
            # print(i, 1 / (2 * self.params[i].data.exp()).item(), self.params[i].data.item())
        
        # loss_sum = loss_sum / 2
        # assert torch.all(loss_sum >= 0), loss_sum[loss_sum < 0]
        return loss_sum
