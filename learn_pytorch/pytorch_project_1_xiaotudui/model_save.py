import torch
import torchvision

vgg_16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 模型结构+模型参数
torch.save(vgg_16,"vgg_16_method1.pth")


# 保存方式2 模型参数（官方推荐）
torch.save(vgg_16.state_dict(), "vgg_16_method2.pth")