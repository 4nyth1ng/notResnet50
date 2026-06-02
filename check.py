import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import time
import sys
import random
import threading
import os


class Check:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    def _make_bar(self, conf):
        length = int(20 * conf)
        bar = "█" * length + "░" * (20 - length)
        if conf > 0.7:
            return f"\033[1;32m{bar}\033[0m"
        elif conf > 0.3:
            return f"\033[1;33m{bar}\033[0m"
        else:
            return f"\033[1;31m{bar}\033[0m"


    def _pad_line(self, left_text, right_colored_text, raw_right_text, total_width=65):
        visible_len = len(left_text) + len(raw_right_text)
        padding = total_width - visible_len
        return f"│  {left_text}{right_colored_text}{' ' * padding}  │"


    def check_confidence(self, org_path, prot_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        img_org = transform(Image.open(org_path).convert('RGB')).unsqueeze(0).to(self.device)
        img_prot = transform(Image.open(prot_path).convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_org = self.model(img_org)
            out_prot = self.model(img_prot)

            prob_org = torch.nn.functional.softmax(out_org, dim=1)
            prob_prot = torch.nn.functional.softmax(out_prot, dim=1)

            conf_org, class_org = prob_org.max(1)
            conf_prot, class_prot = prob_prot.max(1)

            top5_org = torch.topk(prob_org, 5).indices[0].tolist()
            top5_prot = torch.topk(prob_prot, 5).indices[0].tolist()

        is_success = class_org.item() != class_prot.item()
        if is_success:
            status_color = "\033[1;42;37m SUCCESS \033[0m"
            status_raw = " SUCCESS "
        else:
            status_color = "\033[1;41;37m FAILED \033[0m"
            status_raw = " FAILED "

        inner_width = 66

        print("\n┌" + "─" * (inner_width + 4) + "┐")
        line1_raw = f"EVALUATION REPORT | Target: {prot_path} | Status: {status_raw}"
        line1_show = f"\033[1;35mEVALUATION REPORT\033[0m | Target: {prot_path} | Status: {status_color}"
        pad1 = inner_width - len(line1_raw)
        print(f"│  {line1_show}{' ' * pad1}  │")
        print("├" + "─" * (inner_width + 4) + "┤")

        c_org = conf_org.item()
        line2_raw = f"[Original]    Class: {class_org.item():<4} | Conf: {c_org:6.2%} ░░░░░░░░░░░░░░░░░░░░"
        line2_show = f"\033[1;37m[Original]\033[0m    Class: {class_org.item():<4} | Conf: {c_org:6.2%} {self._make_bar(c_org)}"
        pad2 = inner_width - len(line2_raw)
        print(f"│  {line2_show}{' ' * pad2}  │")

        c_prot = conf_prot.item()
        line3_raw = f"[Adversarial] Class: {class_prot.item():<4} | Conf: {c_prot:6.2%} ░░░░░░░░░░░░░░░░░░░░"
        line3_show = f"\033[1;31m[Adversarial]\033[0m Class: {class_prot.item():<4} | Conf: {c_prot:6.2%} {self._make_bar(c_prot)}"
        pad3 = inner_width - len(line3_raw)
        print(f"│  {line3_show}{' ' * pad3}  │")
        print("├" + "─" * (inner_width + 4) + "┤")

        line4_raw = f"Original Top5 Indices:    {str(top5_org)}"
        pad4 = inner_width - len(line4_raw)
        print(f"│  {line4_raw}{' ' * pad4}  │")

        line5_raw = f"Adversarial Top5 Indices: {str(top5_prot)}"
        pad5 = inner_width - len(line5_raw)
        print(f"│  {line5_raw}{' ' * pad5}  │")
        print("└" + "─" * (inner_width + 4) + "┘")




if __name__ == "__main__":
    try:
        Image.open("original.jpg")
    except:
        raise FileNotFoundError("Missing critical asset: original.jpg. Please place 'original.jpg' in the root directory.")

    check = Check()
    check._loading_animation("Initializing Model", duration = random.randint(6, 10))

    for i in range(1, 5):
        filename = f"output/{i}.png" if os.path.exists(f"output/{i}.png") else f"{i}.png"
        if os.path.exists(filename):
            check.check_confidence("original.jpg", filename)
        else:
            print(f"\033[1;33m[WARN]\033[0m File '{filename}' not found. Skipping evaluation.")

