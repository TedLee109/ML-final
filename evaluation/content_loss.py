from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import vgg19
import os

# 定義 VGG 特徵提取器
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name="conv4_4"):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True)
        self.model = nn.Sequential(*list(vgg.features[:26]))
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结 VGG 参数

    def forward(self, x):
        return self.model(x)

# 定義改良 Content Loss
class ImprovedContentLoss(nn.Module):
    def __init__(self):
        super(ImprovedContentLoss, self).__init__()
        self.vgg = VGGFeatureExtractor()  # 初始化特徵提取器
        self.l1_loss = nn.L1Loss()        # 使用 L1 損失

    def forward(self, generated, original):
        # 提取生成圖像與原始圖像的特徵
        generated_features = self.vgg(generated)
        original_features = self.vgg(original)
        # 計算 L1 損失
        loss = self.l1_loss(generated_features, original_features)
        return loss

# 計算 Gram 矩陣
def gram_matrix(features):
    if features.dim() == 3:  # 如果只有 (c, h, w)
        features = features.unsqueeze(0)  # 增加批次維度，變為 (1, c, h, w)

    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # 批次矩陣相乘
    return gram / (c * h * w)

# 定義 Style Loss
class StyleLoss(nn.Module):
    def __init__(self, style_layers=None):
        super(StyleLoss, self).__init__()
        self.vgg = VGGFeatureExtractor()  # 使用同樣的特徵提取器
        self.style_layers = style_layers or ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]  # 默認層
        self.l1_loss = nn.L1Loss()  # 使用 L1 損失

    def forward(self, generated, style):
        generated_features = self.vgg(generated)
        style_features = self.vgg(style)
        
        # 計算每層的 Gram 矩陣並累加損失
        loss = 0
        for gen_feat, style_feat in zip(generated_features, style_features):
            gram_gen = gram_matrix(gen_feat)
            gram_style = gram_matrix(style_feat)
            loss += self.l1_loss(gram_gen, gram_style)
        return loss



# 加載和預處理圖像
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 將圖像調整為 256x256
    transforms.ToTensor(),          # 轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG 的標準化
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化風格損失函數並移動到設備
content_loss_fn = ImprovedContentLoss().to(device)

# 設定宮崎駿風格圖像資料夾
content_image_folder = "origin/images_animal"
content_image_paths = [os.path.join(content_image_folder, fname) for fname in os.listdir(content_image_folder) if fname.endswith((".jpg", ".png"))]

# 設定生成圖片的資料夾
folder = "our_model/outputs_animal/outputs"

# 定義結果字典
average_losses = {}
image_losses = {}  # 新增存储每张图片的损失值的字典

# 對每個資料夾計算平均風格損失

total_style_loss = 0
num_images = 0
i = 0
# 取得資料夾內的圖片路徑
generated_image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith((".jpg", ".png"))]

image_losses[folder] = []  # 初始化該資料夾的損失列表

for generated_image_path, content_image_path in zip(generated_image_paths, content_image_paths):
    # 加載生成圖像並移動到設備
    generated_image = transform(Image.open(generated_image_path).convert("RGB")).unsqueeze(0).to(device)
    if i % 5 == 0:
        print(f"Processing image {i} in folder {folder}")
    i += 1
    # 計算風格損失
    
    content_image = transform(Image.open(content_image_path).convert("RGB")).unsqueeze(0).to(device)
    s_loss = content_loss_fn(generated_image, content_image)
    total_style_loss += s_loss.item()

    num_images += 1
    # 將檔名與損失存入字典
    image_losses[folder].append((os.path.basename(generated_image_path), s_loss.item()))

# 計算資料夾內圖片的平均風格損失
if num_images > 0:
    average_loss = total_style_loss / num_images
    average_losses[folder] = average_loss
else:
    average_losses[folder] = None

# 將結果寫入 txt 檔
output_file = "content_loss_results.txt"
with open(output_file, "w") as f:
    for folder, losses in image_losses.items():
        f.write(f"Folder: {folder}\n")
        for filename, loss in losses:
            f.write(f"Image: {filename}, Style Loss: {loss:.6f}\n")
        if average_losses[folder] is not None:
            f.write(f"Average Style Loss for {folder}: {average_losses[folder]:.6f}\n")
        else:
            f.write("No valid images found in this folder.\n")
        f.write("\n")

print(f"Results have been saved to {output_file}")


