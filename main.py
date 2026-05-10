import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device).eval()

def normal(image_path, output_path):
    img_org = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_org.size

    transform = transforms.ToTensor()
    img_tensor = transform(img_org).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    output = model(F.interpolate(img_tensor, size=(224, 224)))
    target = torch.tensor([output.data.argmax()]).to(device)
    
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    epsilon = 0.1

    grad_sign = F.interpolate(img_tensor.grad.sign(), size=(orig_h, orig_w))

    perturbed_data = img_tensor + epsilon * grad_sign
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    final_img = transforms.ToPILImage()(perturbed_data.squeeze(0).cpu())
    final_img.save(output_path, quality=100)
    print(f"1 {output_path}")
    print(f"1 original size: {orig_w}x{orig_h}")

normal("original.jpg", "1.png")


# ==============================



def pgd(image_path, output_path):
    img_org = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_org.size
    
    img_tensor = transforms.ToTensor()(img_org).unsqueeze(0).to(device)

    iters = 20
    alpha = 0.02
    epsilon = 0.1

    adv_img = img_tensor.clone().detach()

    with torch.no_grad():
        logits = model(F.interpolate(img_tensor, size=(224, 224)))
        target = torch.tensor([logits.argmax()]).to(device)

    for i in range(iters):
        adv_img.requires_grad = True
        outputs = model(F.interpolate(adv_img, size=(224, 224)))
        loss = F.cross_entropy(outputs, target)
        
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_img = adv_img + alpha * adv_img.grad.sign()
            eta = torch.clamp(adv_img - img_tensor, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(img_tensor + eta, min=0, max=1)

        if i % 5 == 0:
            print(f"Iterate {i}/{iters}...")

    final_img = transforms.ToPILImage()(adv_img.squeeze(0).cpu())
    final_img.save(output_path, quality=100)
    print(f"2 {output_path}")

pgd("original.jpg", "2.png")


# ========================================


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

to_tensor = transforms.ToTensor()


def preprocess_224(img):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])(img).unsqueeze(0).to(device)


def deprocess(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)


def pgd(image_path, output_path):
    img_org = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_org.size

    img_small = preprocess_224(img_org)
    img_small_orig = img_small.clone().detach()

    # original
    with torch.no_grad():
        pred = model(img_small).argmax(dim=1)

    # pgd
    iters = 80
    alpha = 2/255
    epsilon = 12/255

    img_adv = img_small.clone().detach()

    for i in range(iters):
        img_adv.requires_grad = True

        outputs = model(img_adv)
        loss = F.cross_entropy(outputs, pred)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            img_adv = img_adv + alpha * img_adv.grad.sign()

            eta = torch.clamp(img_adv - img_small_orig, min=-epsilon, max=epsilon)
            img_adv = img_small_orig + eta

    img_small_clean = deprocess(img_small_orig)
    img_small_adv = deprocess(img_adv)

    noise_small = img_small_adv - img_small_clean

    noise_full = F.interpolate(
        noise_small,
        size=(orig_h, orig_w),
        mode='nearest'
    )

    img_full = to_tensor(img_org).unsqueeze(0).to(device)

    protected = torch.clamp(img_full + noise_full, 0, 1)

    transforms.ToPILImage()(protected.squeeze().cpu()).save(output_path, quality=100)

    print(f"3 {output_path}")

pgd("original.jpg", "3.png")


# =============================================


def upscale(image_path, output_path):
    img_org = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_org.size

    img_small = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])(img_org).unsqueeze(0).to(device)
    img_small.requires_grad = True

    iters = 50
    alpha = 0.00012
    epsilon = 0.02
    
    target = torch.tensor([model(img_small).argmax()]).to(device)

    for i in range(iters):
        img_small.requires_grad = True
        outputs = model(img_small)
        loss = F.cross_entropy(outputs, target)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            img_small = img_small + alpha * img_small.grad.sign()
            eta = torch.clamp(img_small - img_small, min=-epsilon, max=epsilon)
            img_small = torch.clamp(img_small, 0, 1)

    with torch.no_grad():
        img_full = transforms.ToTensor()(img_org).unsqueeze(0).to(device)
        noise_small = img_small - transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img_org).unsqueeze(0).to(device)
        noise_full = F.interpolate(noise_small, size=(orig_h, orig_w), mode='bilinear')

        protected_full = torch.clamp(img_full + noise_full, 0, 1)

    transforms.ToPILImage()(protected_full.squeeze(0).cpu()).save(output_path, quality=100)
    print(f"4 {output_path}")

upscale("original.jpg", "4.png")

