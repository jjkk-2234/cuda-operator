import torch

def batchnorm(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5):

    dims = [d for d in range(x.dim()) if d != 1]
     
    shape = [1, -1] + [1] * (x.dim() - 2)

    if training:
        mean = x.mean(dim=dims, keepdim=True)
        var_biased = x.var(dim=dims, keepdim=True, correction=0)
        var_unbiased = x.var(dim=dims, keepdim=True, correction=1)

        with torch.no_grad():
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean.squeeze()
            running_var.data = (1 - momentum) * running_var.data + momentum * var_unbiased.squeeze()
        var = var_biased

    else:
        mean = running_mean.view(shape)
        var = running_var.view(shape)

    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        w = weight.view(shape)
        b = bias.view(shape)
        x_norm = x_norm * w + b
    return x_norm

if __name__ == "__main__":
    x = torch.randn(2, 3, 4, 5)
    running_mean = torch.zeros(3)
    running_var = torch.ones(3)
    weight = torch.ones(3)
    bias = torch.zeros(3)
    x_norm = batchnorm(x, running_mean, running_var, weight, bias, True)
    print(x_norm)
