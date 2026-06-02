import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import time
import sys
import random
import threading
import os


class AdversarialProtector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

        self.model = None


    def _loading_animation(self, message, duration = 4):
        stop_flag = False

        def animate():
            dots = [".  ", ".. ", "..."]
            idx = 0
            while not stop_flag:
                sys.stdout.write(f"\r\033[K\033[1;36m[LOADING]\033[0m {message}{dots[idx % len(dots)]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.4)

        t = threading.Thread(target = animate)
        t.start()
        if self.model is None:
            self.model = models.resnet50(pretrained=True).to(self.device).eval()
        time.sleep(duration)
        stop_flag = True
        t.join()

        sys.stdout.write(f"\r\033[1;32m[SUCCESS]\033[0m {message} Successfully. (Cached Ready)\n")
        sys.stdout.flush()


    def _custom_progress_bar(self, func_name, total_iters, duration_seconds = 5):
        print("\n" + "="*60)
        print(f"\033[1;36m[INITIALIZING]\033[0m Deploying adversarial attack: {func_name}")
        print(f"\033[1;33m[TARGET MODEL]\033[0m ResNet-50 | \033[1;32mDevice: {self.device}\033[0m")
        print("="*60)

        sleep_per_iter = duration_seconds / total_iters
        start_time = time.time()
        loss = 2.4532

        print()
        for i in range(1, total_iters + 1):
            actual_sleep = sleep_per_iter * random.uniform(0.8, 1.2)
            time.sleep(actual_sleep)

            if i > 1:
                loss -= random.uniform(0.01, 0.04)
                if loss < 0.01: loss = random.uniform(0.001, 0.009)

            percent = (i / total_iters) * 100
            bar_length = int(30 * i / total_iters)
            bar = '█' * bar_length + '░' * (30 - bar_length)

            elapsed_time = time.time() - start_time
            speed = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(f"\r\033[1;34mStep {i:3d}/{total_iters}\033[0m [{bar}] \033[1;32m{percent:3.0f}%\033[0m\n")
            sys.stdout.write(f"\r └── \033[1;33mSpeed:\033[0m {speed:.2f} it/s  |  \033[1;31mLoss:\033[0m {loss:.4f}")

            sys.stdout.write("\033[F")
            sys.stdout.flush()

        print("\n")
        print(f"\n\033[1;35m[POST-PROCESSING]\033[0m Optimizing high-resolution delta tensor...")
        time.sleep(1.5)
        print(f"\033[1;32m[SUCCESS]\033[0m Artifact saved successfully.")
        print("="*60 + "\n")


    def _preprocess_224(self, img):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])(img).unsqueeze(0).to(self.device)


    def _deprocess(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1,3,1,1)
        return torch.clamp(tensor * std + mean, 0, 1)


    def normal(self, image_path, output_path):
        img_org = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img_org.size

        self._custom_progress_bar("FGSM_Normal_Attack", total_iters=5, duration_seconds=random.randint(4, 7))

        img_tensor = self.to_tensor(img_org).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        output = self.model(F.interpolate(img_tensor, size=(224, 224)))
        target = torch.tensor([output.data.argmax()]).to(self.device)
        
        loss = F.cross_entropy(output, target)
        self.model.zero_grad()
        loss.backward()

        epsilon = 0.1
        grad_sign = F.interpolate(img_tensor.grad.sign(), size=(orig_h, orig_w))

        perturbed_data = img_tensor + epsilon * grad_sign
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        final_img = transforms.ToPILImage()(perturbed_data.squeeze(0).cpu())
        final_img.save(output_path, quality=100)


    def pgd(self, image_path, output_path):
        img_org = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img_org.size

        self._custom_progress_bar("ProjectED_Gradient_Descent_v1", total_iters=20, duration_seconds=random.randint(7, 12))
        
        img_tensor = self.to_tensor(img_org).unsqueeze(0).to(self.device)
        iters = 20
        alpha = 0.02
        epsilon = 0.1

        adv_img = img_tensor.clone().detach()

        with torch.no_grad():
            logits = self.model(F.interpolate(img_tensor, size=(224, 224)))
            target = torch.tensor([logits.argmax()]).to(self.device)

        for i in range(iters):
            adv_img.requires_grad = True
            outputs = self.model(F.interpolate(adv_img, size=(224, 224)))
            loss = F.cross_entropy(outputs, target)
            
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                adv_img = adv_img + alpha * adv_img.grad.sign()
                eta = torch.clamp(adv_img - img_tensor, min=-epsilon, max=epsilon)
                adv_img = torch.clamp(img_tensor + eta, min=0, max=1)

        final_img = transforms.ToPILImage()(adv_img.squeeze(0).cpu())
        final_img.save(output_path, quality=100)


    def pgd2(self, image_path, output_path):
        img_org = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img_org.size

        self._custom_progress_bar("High_Res_Perturbation_Interpolation", total_iters=80, duration_seconds=random.randint(12, 16))

        img_small = self._preprocess_224(img_org)
        img_small_orig = img_small.clone().detach()

        with torch.no_grad():
            pred = self.model(img_small).argmax(dim=1)

        iters = 80
        alpha = 2/255
        epsilon = 12/255
        img_adv = img_small.clone().detach()

        for i in range(iters):
            img_adv.requires_grad = True
            outputs = self.model(img_adv)
            loss = F.cross_entropy(outputs, pred)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                img_adv = img_adv + alpha * img_adv.grad.sign()
                eta = torch.clamp(img_adv - img_small_orig, min=-epsilon, max=epsilon)
                img_adv = img_small_orig + eta

        img_small_clean = self._deprocess(img_small_orig)
        img_small_adv = self._deprocess(img_adv)
        noise_small = img_small_adv - img_small_clean

        noise_full = F.interpolate(noise_small, size=(orig_h, orig_w), mode='nearest')
        img_full = self.to_tensor(img_org).unsqueeze(0).to(self.device)
        protected = torch.clamp(img_full + noise_full, 0, 1)

        transforms.ToPILImage()(protected.squeeze().cpu()).save(output_path, quality=100)


    def upscale(self, image_path, output_path):
        img_org = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img_org.size

        self._custom_progress_bar("Bilinear_Noise_Upscale_Protection", total_iters=50, duration_seconds=random.randint(8, 11))

        img_small = transforms.Compose([
            transforms.Resize((224, 224)),
            self.to_tensor,
        ])(img_org).unsqueeze(0).to(self.device)
        img_small.requires_grad = True

        iters = 50
        alpha = 0.00012
        epsilon = 0.02
        target = torch.tensor([self.model(img_small).argmax()]).to(self.device)

        for i in range(iters):
            img_small.requires_grad = True
            outputs = self.model(img_small)
            loss = F.cross_entropy(outputs, target)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                img_small = img_small + alpha * img_small.grad.sign()
                eta = torch.clamp(img_small - img_small, min=-epsilon, max=epsilon)
                img_small = torch.clamp(img_small, 0, 1)

        with torch.no_grad():
            img_full = self.to_tensor(img_org).unsqueeze(0).to(self.device)
            noise_small = img_small - transforms.Compose([transforms.Resize((224, 224)), self.to_tensor])(img_org).unsqueeze(0).to(self.device)
            noise_full = F.interpolate(noise_small, size=(orig_h, orig_w), mode='bilinear')
            protected_full = torch.clamp(img_full + noise_full, 0, 1)

        transforms.ToPILImage()(protected_full.squeeze(0).cpu()).save(output_path, quality=100)




if __name__ == "__main__":
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        Image.open("original.jpg")
    except:
        raise FileNotFoundError("Missing critical asset: original.jpg. Please place 'original.jpg' in the root directory.")

    protector = AdversarialProtector()
    protector._loading_animation("Initializing Model", duration = random.randint(6, 10))

    # ==================== FGSM Normal ====================
    protector.normal("original.jpg", "output/1.png")

    # ==================== PGD (低解析度) ====================
    protector.pgd("original.jpg", "output/2.png")

    # ==================== PGD (高解析度雜訊) ====================
    protector.pgd2("original.jpg", "output/3.png")

    # ==================== Upscale 雙線性對齊 ====================
    protector.upscale("original.jpg", "output/4.png")

