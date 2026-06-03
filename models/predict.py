"""
CNN 推理预测模块
加载训练好的模型权重，对单张 / 批量图像进行品种识别
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from models.train import build_model

# 品种中文名称映射
BREED_CN = {
    "Abyssinian": "阿比西尼亚猫", "American Bulldog": "美国斗牛犬", "American Pit Bull Terrier": "美国比特犬",
    "Basset Hound": "巴吉度猎犬", "Beagle": "比格犬", "Bengal": "孟加拉猫", "Birman": "伯曼猫",
    "Bombay": "孟买猫", "Boxer": "拳师犬", "British Shorthair": "英国短毛猫", "Chihuahua": "吉娃娃",
    "Egyptian Mau": "埃及猫", "English Cocker Spaniel": "英国可卡犬", "English Setter": "英国雪达犬",
    "German Shorthaired": "德国短毛指示犬", "Great Pyrenees": "大白熊犬", "Havanese": "哈瓦那犬",
    "Japanese Chin": "日本狆", "Keeshond": "荷兰毛狮犬", "Leonberger": "莱昂贝格犬",
    "Maine Coon": "缅因猫", "Miniature Pinscher": "迷你品犬", "Newfoundland": "纽芬兰犬",
    "Persian": "波斯猫", "Pomeranian": "博美犬", "Pug": "巴哥犬", "Ragdoll": "布偶猫",
    "Russian Blue": "俄罗斯蓝猫", "Saint Bernard": "圣伯纳犬", "Samoyed": "萨摩耶犬",
    "Scottish Terrier": "苏格兰梗", "Shiba Inu": "柴犬", "Siamese": "暹罗猫", "Sphynx": "斯芬克斯猫",
    "Staffordshire Bull Terrier": "斯塔福郡斗牛梗", "Wheaten Terrier": "爱尔兰软毛梗",
    "Yorkshire Terrier": "约克夏梗"
}



# ImageNet 1000类中猫狗品种索引 → 品种名映射（无微调模型时使用）
IMAGENET_PET_MAP = {
    151: ("Chihuahua", "吉娃娃犬"),
    152: ("Japanese Chin", "日本狆"),
    161: ("Basset Hound", "巴吉度猎犬"),
    162: ("Beagle", "比格犬"),
    167: ("English Foxhound", "英国猎狐犬"),
    179: ("Staffordshire Bull Terrier", "斯塔福郡斗牛梗"),
    180: ("American Staffordshire Terrier", "美国斯塔福郡梗"),
    181: ("Bedlington Terrier", "贝灵顿梗"),
    182: ("Border Terrier", "边境梗"),
    187: ("Yorkshire Terrier", "约克夏梗"),
    195: ("Boston Terrier", "波士顿梗"),
    202: ("Wheaten Terrier", "爱尔兰软毛梗"),
    205: ("Flat-coated Retriever", "平毛寻回犬"),
    207: ("Golden Retriever", "金毛寻回犬"),
    208: ("Labrador Retriever", "拉布拉多犬"),
    209: ("Chesapeake Bay Retriever", "切萨皮克湾寻回犬"),
    212: ("English Setter", "英国雪达犬"),
    213: ("Irish Setter", "爱尔兰雪达犬"),
    232: ("Border Collie", "边境牧羊犬"),
    234: ("Rottweiler", "罗威纳犬"),
    235: ("German Shepherd", "德国牧羊犬"),
    237: ("Miniature Pinscher", "迷你品犬"),
    242: ("Boxer", "拳师犬"),
    245: ("French Bulldog", "法国斗牛犬"),
    246: ("Great Dane", "大丹犬"),
    247: ("Saint Bernard", "圣伯纳犬"),
    250: ("Siberian Husky", "西伯利亚哈士奇"),
    251: ("Dalmatian", "斑点犬"),
    254: ("Pug", "巴哥犬"),
    255: ("Leonberger", "莱昂贝格犬"),
    256: ("Newfoundland", "纽芬兰犬"),
    257: ("Great Pyrenees", "大白熊犬"),
    258: ("Samoyed", "萨摩耶犬"),
    259: ("Pomeranian", "博美犬"),
    261: ("Keeshond", "荷兰毛狮犬"),
    281: ("Tabby Cat", "虎斑猫"),
    283: ("Persian Cat", "波斯猫"),
    284: ("Siamese Cat", "暹罗猫"),
    285: ("Egyptian Mau", "埃及猫"),
}

class PetBreedClassifier:
    """宠物品种分类器"""

    def __init__(self, model_path="weights/pet_breed_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # 加载模型
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.class_names = checkpoint["class_names"]
            self.model = build_model(num_classes=len(self.class_names), pretrained=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._is_fine_tuned = True
            print(f"模型已加载 (准确率: {checkpoint.get('best_acc', 'N/A')})")
        else:
            # 未找到训练权重时使用 ImageNet 预训练模型 + 猫狗品种映射
            print("未找到微调模型，使用 ImageNet 预训练模型 + 猫狗品种映射作为演示")
            from torchvision import models
            self.model = models.resnet18(weights="DEFAULT")
            self.class_names = []
            self._is_fine_tuned = False

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        对单张图片进行预测
        :param image: PIL.Image 或 图像路径
        :return: dict {"breed": 品种名, "breed_cn": 中文名, "confidence": 置信度}
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)

        # 取 Top-10 预测索引
        top10_probs, top10_indices = probs.topk(min(10, probs.size(1)))

        if self._is_fine_tuned:
            # 微调模型：class_names 是品种名字符串列表
            pred_idx = top10_indices[0, 0].item()
            confidence = top10_probs[0, 0].item()
            breed = self.class_names[pred_idx]
            breed_cn = BREED_CN.get(breed, breed)
        else:
            # ImageNet 回退：从 Top-10 中找匹配的猫狗品种
            matched = None
            matched_confidence = 0
            for i in range(len(top10_indices[0])):
                idx_val = top10_indices[0, i].item()
                if idx_val in IMAGENET_PET_MAP:
                    matched = IMAGENET_PET_MAP[idx_val]
                    matched_confidence = top10_probs[0, i].item()
                    break
            if matched:
                breed, breed_cn = matched
                confidence = matched_confidence
            else:
                breed = "未知品种"
                breed_cn = "未知品种"
                confidence = 0.0

        return {
            "breed": breed,
            "breed_cn": breed_cn,
            "confidence": round(confidence * 100, 2),
            "top5": self._get_top5(probs)
        }

    def _get_top5(self, probs):
        """获取 Top-5 预测结果"""
        top5_prob, top5_idx = probs.topk(min(5, probs.size(1)))
        results = []
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            idx_val = idx.item()
            if self._is_fine_tuned:
                breed = self.class_names[idx_val] if idx_val < len(self.class_names) else f"类别_{idx_val}"
                breed_cn = BREED_CN.get(breed, breed)
            elif idx_val in IMAGENET_PET_MAP:
                breed, breed_cn = IMAGENET_PET_MAP[idx_val]
            else:
                breed = f"类别_{idx_val}"
                breed_cn = breed
            results.append({
                "breed": breed,
                "breed_cn": breed_cn,
                "confidence": round(prob.item() * 100, 2)
            })
        return results


if __name__ == "__main__":
    classifier = PetBreedClassifier()
    import sys
    if len(sys.argv) > 1:
        result = classifier.predict(sys.argv[1])
        print(f"\n预测结果: {result['breed_cn']} ({result['breed']})")
        print(f"置信度: {result['confidence']}%")
        print("\nTop-5:")
        for r in result["top5"]:
            print(f"  {r['breed_cn']:12s}  {r['confidence']:6.2f}%")
