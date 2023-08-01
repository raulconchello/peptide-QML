import torch

class QNodeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, qlayer):
        ctx.save_for_backward(input)
        ctx.qlayer = qlayer
        return qlayer(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        qlayer = ctx.qlayer
        epsilon = 0.1
        
        # Gradient w.r.t. input
        perturbation_input = (torch.rand_like(input) * 2 - 1)
        positive_input = input + perturbation_input * epsilon
        negative_input = input - perturbation_input * epsilon
        
        loss_positive = qlayer(positive_input).sum()
        loss_negative = qlayer(negative_input).sum()
        
        gradient_input = (loss_positive - loss_negative) / (2 * epsilon) * perturbation_input
        gradient_input *= grad_output  # Incorporate grad_output due to chain rule

        # Gradient w.r.t. qlayer's parameters
        gradients_weights = []
        for p in qlayer.parameters():
            perturbation_weight = (torch.rand_like(p) * 2 - 1) 

            p.data += perturbation_weight * epsilon  # Apply positive perturbation
            loss_positive = qlayer(input).sum()

            p.data -= 2*perturbation_weight * epsilon  # Apply negative perturbation
            loss_negative = qlayer(input).sum()

            gradient_weight = (loss_positive - loss_negative) / (2 * epsilon) * perturbation_weight 
            gradients_weights.append(gradient_weight * grad_output.sum())  # Weighting by grad_output

            p.data += perturbation_weight * epsilon  # Revert parameter back to original value

        # Update gradients for qlayer's parameters
        for p, grad in zip(qlayer.parameters(), gradients_weights):
            if p.grad is None:
                p.grad = grad.detach()
            else:
                p.grad += grad.detach()

        return gradient_input, None

# Wrapper around the custom autograd function
class QLayer(torch.nn.Module):
    def __init__(self, qlayer):
        super(QLayer, self).__init__()
        self.qlayer = qlayer

    def forward(self, x):
        return QNodeFunction.apply(x, self.qlayer)