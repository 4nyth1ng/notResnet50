import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device).eval()


def check_ai_confidence(org_path, prot_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_org = transform(Image.open(org_path).convert('RGB')).unsqueeze(0).to(device)
    img_prot = transform(Image.open(prot_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        out_org = model(img_org)
        out_prot = model(img_prot)

        prob_org = torch.nn.functional.softmax(out_org, dim=1)
        prob_prot = torch.nn.functional.softmax(out_prot, dim=1)

        conf_org, class_org = prob_org.max(1)
        conf_prot, class_prot = prob_prot.max(1)

    print("="*50)
    print(f"original: class={class_org.item()}, conf={conf_org.item():.2%}")
    print(f"adversarial: class={class_prot.item()}, conf={conf_prot.item():.2%}")

    top5_org = torch.topk(prob_org, 5)
    top5_prot = torch.topk(prob_prot, 5)

    print("\noriginal Top5:", top5_org.indices)
    print("adversarial Top5:", top5_prot.indices)

    print()
    if class_org != class_prot:
        print("success")
    else:
        print("fail")



check_ai_confidence("original.jpg", "1.png")


check_ai_confidence("original.jpg", "2.png")


check_ai_confidence("original.jpg", "3.png")


check_ai_confidence("original.jpg", "4.png")


