import torch
import torchvision.transforms
from PIL import Image
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("训练的设备是：{}".format(device))

image_path = "./test_img/dog.png"
image = Image.open(image_path)
print(image) # PIL格式

# 调整照片大小，并转换为tensor类型数据
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image =transform(image)
image = torch.reshape(image,(1,3,32,32))
print(image.shape)

model =torch.load("./pretrained_model/model50.pth")
print(model)

model.eval()
with torch.no_grad():
    output = model(image.to(device))

print(output.argmax(1))