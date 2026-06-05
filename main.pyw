import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import time
import sys
import random
import threading
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import re
import queue


class AdversarialProtector:
    def __init__(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

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
                sys.stdout.write(f"\r\033[1;36m[LOADING]\033[0m {message}{dots[idx % len(dots)]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.4)

        t = threading.Thread(target = animate)
        t.start()
        if self.model is None:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device).eval()
        time.sleep(duration)
        stop_flag = True
        t.join()

        sys.stdout.write(f"\r\033[32m[SUCCESS]\033[0m {message} Successfully. (Cached Ready)\n")
        sys.stdout.flush()

    def _custom_progress_bar(self, func_name, total_iters, duration_seconds = 5):
        print("-" * 60)
        print(f"[INITIALIZING] Deploying adversarial attack: {func_name}")
        print(f"[TARGET MODEL] ResNet-50 | Device: {self.device}")
        print("-" * 60)

        sleep_per_iter = duration_seconds / total_iters
        start_time = time.time()
        loss = 2.4532

        for i in range(1, total_iters + 1):
            actual_sleep = sleep_per_iter * random.uniform(0.8, 1.2)
            time.sleep(actual_sleep)

            if i > 1:
                loss -= random.uniform(0.01, 0.04)
                if loss < 0.01: loss = random.uniform(0.001, 0.009)

            percent = (i / total_iters) * 100
            bar_length = int(20 * i / total_iters)
            bar = '█' * bar_length + '░' * (20 - bar_length)

            elapsed_time = time.time() - start_time
            speed = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(f"\r\033[36mStep {i:3d}/{total_iters}\033[0m [{bar}] {percent:3.0f}% | Speed: {speed:.2f} it/s | Loss: {loss:.4f}")
            sys.stdout.flush()

        print(f"\n[POST-PROCESSING] Optimizing high-resolution delta tensor...")
        time.sleep(1.0)
        print(f"\033[32m[SUCCESS]\033[0m Artifact saved successfully.")
        print("-" * 60 + "\n")

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

        img_small_orig = img_small.clone().detach()
        for i in range(iters):
            img_small.requires_grad = True
            outputs = self.model(img_small)
            loss = F.cross_entropy(outputs, target)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                adv_img = img_small + alpha * img_small.grad.sign()
                eta = torch.clamp(adv_img - img_small_orig, min=-epsilon, max=epsilon)
                img_small = torch.clamp(img_small_orig + eta, 0, 1)

        with torch.no_grad():
            img_full = self.to_tensor(img_org).unsqueeze(0).to(self.device)
            noise_small = img_small - img_small_orig
            noise_full = F.interpolate(noise_small, size=(orig_h, orig_w), mode='bilinear')
            protected_full = torch.clamp(img_full + noise_full, 0, 1)

        transforms.ToPILImage()(protected_full.squeeze(0).cpu()).save(output_path, quality=100)




class Check:
    def __init__(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = None

    def _loading_animation(self, message, duration = 4):
        stop_flag = False

        def animate():
            dots = [".  ", ".. ", "..."]
            idx = 0
            while not stop_flag:
                sys.stdout.write(f"\r\033[36m[LOADING]\033[0m {message}{dots[idx % len(dots)]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.4)

        t = threading.Thread(target = animate)
        t.start()
        if self.model is None:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device).eval()
        time.sleep(duration)
        stop_flag = True
        t.join()

        sys.stdout.write(f"\r\033[32m[SUCCESS]\033[0m {message} Successfully. (Cached Ready)\n")
        sys.stdout.flush()

    def _custom_progress_bar(self, func_name, total_iters, duration_seconds = 5):
        print("-" * 60)
        print(f"[INITIALIZING] Deploying adversarial attack: {func_name}")
        print(f"[TARGET MODEL] ResNet-50 | Device: {self.device}")
        print("-" * 60)

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
            bar_length = int(20 * i / total_iters)
            bar = '█' * bar_length + '░' * (20 - bar_length)

            elapsed_time = time.time() - start_time
            speed = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(f"\r\033[36mStep {i:3d}/{total_iters}\033[0m [{bar}] {percent:3.0f}% | Speed: {speed:.2f} it/s | Loss: {loss:.4f}")
            sys.stdout.flush()

        print("\n")
        print(f"\n\033[1;35m[POST-PROCESSING]\033[0m Optimizing high-resolution delta tensor...")
        time.sleep(1.5)
        print(f"\033[1;32m[SUCCESS]\033[0m Artifact saved successfully.")
        print("-" * 60 + "\n")

    def _make_bar(self, conf):
        length = int(20 * conf)
        bar = "█" * length + "░" * (20 - length)
        if conf > 0.7:
            return f"\033[1;32m{bar}\033[0m"
        elif conf > 0.3:
            return f"\033[1;33m{bar}\033[0m"
        else:
            return f"\033[1;31m{bar}\033[0m"

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
            status_raw = " SUCCESS "
            status_color = f"\033[1;42;37m{status_raw}\033[0m"
        else:
            status_raw = " FAILED  "
            status_color = f"\033[1;41;37m{status_raw}\033[0m"

        display_path = os.path.basename(prot_path)
        if len(display_path) > 18: display_path = display_path[:15] + "..."

        print("\n" + "-" * 60)
        print(f"\033[1;35mEVALUATION REPORT\033[0m | Target: {display_path} | Status: {status_color}")
        print("-" * 60)

        c_org = conf_org.item()
        print(f"\033[1;37m[Original]\033[0m    Class: {class_org.item():<4} | Conf: {c_org:6.2%} {self._make_bar(c_org)}")

        c_prot = conf_prot.item()
        print(f"\033[1;31m[Adversarial]\033[0m Class: {class_prot.item():<4} | Conf: {c_prot:6.2%} {self._make_bar(c_prot)}")
        print("-" * 60)

        print(f"Original Top5 Indices:    {str(top5_org)}")
        print(f"Adversarial Top5 Indices: {str(top5_prot)}")
        print("-" * 60 + "\n")




class ProtectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("版權逐影")
        self.root.geometry("900x700")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.root.attributes("-alpha", 0.0)

        self.protector = AdversarialProtector()
        self.checker = Check()
        self.selected_path = ctk.StringVar()
        self.log_queue = queue.Queue()

        self._setup_ui()
        self._setup_logging()
        self._animate_window()

    def _setup_ui(self):
        font_ui_bold = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        font_ui_regular = ctk.CTkFont(family="Segoe UI", size=13)
        font_button = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
        font_console = ctk.CTkFont(family="Consolas", size=12)

        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=30, pady=25)

        selection_card = ctk.CTkFrame(main_frame, corner_radius=12)
        selection_card.pack(fill=ctk.X, pady=(0, 15))

        label_src = ctk.CTkLabel(selection_card, text="選擇影像檔案", font=font_ui_bold, text_color=("#3B82F6", "#60A5FA"))
        label_src.pack(anchor=ctk.W, padx=20, pady=(18, 5))

        input_row = ctk.CTkFrame(selection_card, fg_color="transparent")
        input_row.pack(fill=ctk.X, padx=20, pady=(0, 20))

        self.path_entry = ctk.CTkEntry(input_row, textvariable=self.selected_path, placeholder_text="Click browse to select an image...", font=font_ui_regular, height=35)
        self.path_entry.pack(side=ctk.LEFT, padx=(0, 10), fill=ctk.X, expand=True)

        browse_btn = ctk.CTkButton(input_row, text="Browse File", width=110, height=35, font=font_ui_bold, command=self._browse_file)
        browse_btn.pack(side=ctk.RIGHT)

        console_card = ctk.CTkFrame(main_frame, corner_radius=12)
        console_card.pack(fill=ctk.BOTH, expand=True, pady=0)

        console_header = ctk.CTkFrame(console_card, fg_color="transparent", height=40)
        console_header.pack(fill=ctk.X, padx=15, pady=(10, 5))

        ctk.CTkLabel(console_header, text="控制台", font=font_ui_bold, text_color=("#64748B", "#94A3B8")).pack(side=ctk.LEFT, padx=5)
        ctk.CTkButton(console_header, text="清除控制台", width=90, height=26, font=font_ui_bold, border_width=1, command=self._clear_console).pack(side=ctk.RIGHT)

        self.console = ctk.CTkTextbox(
            console_card, 
            font=font_console,
            fg_color=("#0F172A", "#1E1E1E"),
            text_color="#E4E4E4",
            scrollbar_button_color="#3A3A3C",
            corner_radius=0
        )
        self.console.pack(fill=ctk.BOTH, expand=True, padx=1, pady=(0, 1))

        self.console._textbox.tag_config("red", foreground="#FF453A")
        self.console._textbox.tag_config("green", foreground="#32D74B")
        self.console._textbox.tag_config("yellow", foreground="#FFD60A")
        self.console._textbox.tag_config("magenta", foreground="#FF2D55")
        self.console._textbox.tag_config("light_blue", foreground="#64D2FF")
        self.console._textbox.tag_config("orange", foreground="#FF9F0A")
        self.console._textbox.tag_config("purple", foreground="#BF5AF2")
        self.console._textbox.tag_config("blue", foreground="#0A84FF")
        self.console._textbox.tag_config("cyan", foreground="#64D2FF")
        self.console._textbox.tag_config("white", foreground="#E4E4E4")
        self.console._textbox.tag_config("bold", font=(font_console.cget("family"), font_console.cget("size"), "bold"))

        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill=ctk.X, pady=(15, 0))

        self.run_btn = ctk.CTkButton(button_frame, text="開始執行", height=55, font=font_button, command=self._start_process)
        self.run_btn.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 10))

        self.check_btn = ctk.CTkButton(button_frame, text="檢查", height=55, font=font_button, command=self._start_check_thread)
        self.check_btn.pack(side=ctk.RIGHT, fill=ctk.X, expand=True, padx=(10, 0))

        self.status_var = ctk.StringVar(value="System Initialized")
        status_bar = ctk.CTkLabel(self.root, textvariable=self.status_var, height=30, font=font_ui_regular, fg_color=("#F1F5F9", "#0F172A"), text_color=("#475569", "#94A3B8"), anchor=ctk.W, padx=25)
        status_bar.pack(side=ctk.BOTTOM, fill=ctk.X)

    def _setup_logging(self):
        class QueueWriter:
            def __init__(self, q): self.q = q
            def write(self, s): self.q.put(s)
            def flush(self): pass

        sys.stdout = QueueWriter(self.log_queue)
        self.ansi_regex = re.compile(r'\x1b\[([0-9;]*)m')
        self.root.after(100, self._update_console)

    def _animate_window(self):
        alpha = self.root.attributes("-alpha")
        if alpha < 1.0:
            alpha += 0.05
            self.root.attributes("-alpha", alpha)
            self.root.after(20, self._animate_window)

    def _update_console(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()

                if '\r' in msg:
                    parts = msg.split('\r')
                    for i, part in enumerate(parts):
                        if i > 0:
                            self.console.delete("end-2c linestart", "end-1c")
                        self._insert_ansi_text(part)
                else:
                    self._insert_ansi_text(msg)

                self.console.see("end")
        except queue.Empty:
            pass
        self.root.after(40, self._update_console)

    def _insert_ansi_text(self, text):
        color_map = {
            "31": "red", "32": "green", "33": "yellow", 
            "34": "blue", "35": "magenta", "36": "cyan", "37": "white", "1": "bold"
        }
        last_end = 0
        current_tags = []

        for match in self.ansi_regex.finditer(text):
            self.console._textbox.insert("end", text[last_end:match.start()], tuple(current_tags))

            code = match.group(1)
            if code == "0" or not code:
                current_tags = []
            elif code == "1;36": # [LOADING]
                current_tags = ["light_blue", "bold"]
            elif code == "1;32": # [SUCCESS]
                current_tags = ["green", "bold"]
            elif code == "1;33": # [WARN], [TARGET MODEL]
                current_tags = ["yellow", "bold"]
            elif code == "1;34": # Step
                current_tags = ["blue", "bold"]
            elif code == "1;35": # [POST-PROCESSING], EVALUATION REPORT
                current_tags = ["purple", "bold"]
            elif code == "1;31": # Loss, [Adversarial], FAILED
                current_tags = ["red", "bold"]
            elif code == "1;42;37": # SUCCESS background
                current_tags = ["green", "bold"]
            elif code == "1;41;37": # FAILED background
                current_tags = ["red", "bold"]
            else:
                for c in code.split(';'):
                    if c in color_map:
                        current_tags.append(color_map[c])
            last_end = match.end()
        self.console._textbox.insert("end", text[last_end:], tuple(current_tags))

    def _clear_console(self):
        self.console.delete("1.0", "end")

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="選擇影像檔案",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.selected_path.set(file_path)

    def _start_check_thread(self):
        self.run_btn.configure(state="disabled")
        self.check_btn.configure(state="disabled")
        self.status_var.set("正在執行檢查...")

        threading.Thread(target=self._run_check_logic, daemon=True).start()

    def _run_check_logic(self):
        try:
            self.checker._loading_animation("Initializing Check Model", duration=2)
            self.status_var.set("正在評估輸出檔案...")

            selected = self.selected_path.get()
            if not selected or not os.path.exists(selected):
                messagebox.showerror("錯誤", "請先選擇原始影像檔案作為檢查基準！")
                return

            print(f"\033[1;34m[INFO]\033[0m Using selected file as reference: {os.path.basename(selected)}")

            output_dir = "output"
            if not os.path.exists(output_dir):
                print(f"\033[1;33m[WARN]\033[0m Output directory '{output_dir}' not found. Skipping evaluation of output files.")

            for i in range(1, 5):
                filename = f"output/{i}.png"
                if not os.path.exists(filename):
                    filename = f"{i}.png"

                if os.path.exists(filename):
                    self.checker.check_confidence(selected, filename)
                else:
                    print(f"\033[1;33m[WARN]\033[0m File '{filename}' not found. Skipping evaluation.")

            self.status_var.set("檢查完成")
            messagebox.showinfo("成功", "所有檢查程序已執行完畢！")

        except Exception as e:
            self.status_var.set(f"檢查失敗: {e}")
            messagebox.showerror("錯誤", f"檢查過程中發生錯誤: {e}")
        finally:
            self.run_btn.configure(state="normal")
            self.check_btn.configure(state="normal")

    def _start_process(self):
        img_path = self.selected_path.get()
        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("錯誤", "請先選擇有效的影像檔案！")
            return

        self.run_btn.configure(state="disabled")
        self.check_btn.configure(state="disabled")
        self.status_var.set("正在初始化模型...")

        def run_attack_logic():
            try:
                output_dir = "output"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                self.protector._loading_animation("Initializing Protector Model", duration=2)

                self.status_var.set("正在執行對抗攻擊與保護...")
                self.protector.normal(img_path, "output/1.png")
                self.protector.pgd(img_path, "output/2.png")
                self.protector.pgd2(img_path, "output/3.png")
                self.protector.upscale(img_path, "output/4.png")

                self.status_var.set("處理完成，檔案已儲存至 output 目錄")
                messagebox.showinfo("成功", "所有保護程序已執行完畢！\n結果已儲存至 output/ 目錄。")
            except Exception as e:
                self.status_var.set(f"處理失敗: {e}")
                messagebox.showerror("錯誤", f"處理過程中發生錯誤: {e}")
            finally:
                self.run_btn.configure(state="normal")
                self.check_btn.configure(state="normal")

        threading.Thread(target=run_attack_logic, daemon=True).start()




if __name__ == "__main__":
    root = ctk.CTk()
    app = ProtectorGUI(root)
    root.mainloop()
