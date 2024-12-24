import os
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import vgg19


# 定義 VGG 特徵提取器
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_names = ['0', '5', '10', '21', '30']  # conv1_1, conv2_1, conv3_1, conv4_2, conv5_2
        vgg = vgg19(pretrained=True)
        self.model = nn.Sequential(*list(vgg.features[:31]))

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.layer_names:
                features.append(x)
        return features


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


# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載和預處理圖像
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 將圖像調整為 256x256
    transforms.ToTensor(),          # 轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG 的標準化
])

# 初始化風格損失函數並移動到設備
style_loss_fn = StyleLoss().to(device)

# 設定宮崎駿風格圖像資料夾
style_image_folder = "miyazaki_images"
style_image_paths = [os.path.join(style_image_folder, fname) for fname in os.listdir(style_image_folder) if fname.endswith((".jpg", ".png"))]

# 設定生成圖片的資料夾
generated_folders = ["our_model/outputs_animal/outputs"]

# 定義結果字典
average_losses = {}
image_losses = {}  # 新增存储每张图片的损失值的字典

# 對每個資料夾計算平均風格損失
for folder in generated_folders:
    total_style_loss = 0
    num_images = 0
    i = 0
    # 取得資料夾內的圖片路徑
    generated_image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith((".jpg", ".png"))]

    image_losses[folder] = []  # 初始化該資料夾的損失列表

    for generated_image_path in generated_image_paths:
        # 加載生成圖像並移動到設備
        generated_image = transform(Image.open(generated_image_path).convert("RGB")).unsqueeze(0).to(device)
        if i % 5 == 0:
            print(f"Processing image {i} in folder {folder}")
        i += 1
        # 計算風格損失
        for style_image_path in style_image_paths:
            # 加載風格圖像並移動到設備
            style_image = transform(Image.open(style_image_path).convert("RGB")).unsqueeze(0).to(device)
            s_loss = style_loss_fn(generated_image, style_image)
            total_style_loss += s_loss.item()

        num_images += 1
        # 將檔名與損失存入字典
        image_losses[folder].append((os.path.basename(generated_image_path), s_loss.item()))

    # 計算資料夾內圖片的平均風格損失
    if num_images > 0:
        average_loss = total_style_loss / num_images
        average_losses[folder] = average_loss * 10
    else:
        average_losses[folder] = None

# 將結果寫入 txt 檔
output_file = "style_loss_results.txt"
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
