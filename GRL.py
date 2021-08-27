from torch.autograd import Function


# 定义GRL
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 定义绝缘层
class Iso_layer(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.alpha = 0
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None