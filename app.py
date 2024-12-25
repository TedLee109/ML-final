import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tkinter import Tk, Label, Button, Entry, StringVar, Scale, HORIZONTAL, OptionMenu
from threading import Thread
import time
from lora_diffusion import patch_pipe, tune_lora_scale

#merged_model_path = "./output_merged"
pic_path = "origin_set/woman2.jpg"
class Miyazaki_hayao_style_transfer:
    def __init__(self, ControlWindow):
        #define the window
        self.ControlWindow = ControlWindow
        self.ControlWindow.title("Miyazaki Hayao Style Transfer")
        self.ControlWindow.geometry("500x500")
        self.ControlWindow.configure(bg='white')
        self.ControlWindow.resizable(False, False)

        # Create an entry for the prompt
        self.prompt_var = StringVar()
        self.prompt_entry = Entry(ControlWindow, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack()
        self.prompt_entry.insert(0, "style of <s1><s2>,a portrait of a man")

        self.cap = None
        self.camera_idx = 0
        self.camera_list = self.get_available_camera_list()
        self.camera_idx_var = StringVar(ControlWindow)
        self.camera_idx_var.set(self.camera_list[0])
        self.camera_menu = OptionMenu(ControlWindow, self.camera_idx_var, *self.camera_list)
        self.camera_menu.pack()

        self.open_camera_button = Button(ControlWindow, text="Start", command=self.startCamera)
        self.open_camera_button.pack()

        self.video_label = Label(ControlWindow)
        self.video_label.pack()

        # Create a slider for style transfer strength
        self.strength_var = StringVar()
        self.strength_scale = Scale(ControlWindow, from_=0.1, to=1.0, resolution=0.05, orient=HORIZONTAL,
                                    label="Style Transfer Strength", variable=self.strength_var)
        self.strength_scale.set(0.75) #default 0.65
        self.strength_scale.pack()

        # the button to take a picture
        self.take_picture_button = Button(ControlWindow, text="Take Picture", command=self.take_pic)
        self.take_picture_button.pack()
        # the button to stop the live generation
        self.close_button = Button(ControlWindow, text="Close", command=self.close)
        self.close_button.pack()

        self.NowScreen = None

        # Load the Stable Diffusion model
        print("Loading stable diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        patch_pipe(self.pipe, "final_lora.safetensors", patch_text=True, patch_unet=True, patch_ti=True)
        torch.manual_seed(1)
        # self.pipe = self.pipe.to("cuda")
        print("Model loading complete! ")
        print(f"device at:{self.pipe.device}")

        # Start the video feed
        self.showing_video()

    def close(self):
        if self.cap != None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.ControlWindow.quit()

    def get_available_camera_list(self):

        camera_list = []
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(f"Camera {i}")
                cap.release()
        return camera_list      

    def startCamera(self):
        self.camera_idx = int(self.camera_idx_var.get().split(" ")[1])

        if self.cap != None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(self.camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
    def showing_video(self):

        if self.cap is not None:
            ret, Screen = self.cap.read()
            if ret:
                
                # 水平翻轉畫面
                flipped_screen = cv2.flip(Screen, 1)  # 1 表示左右翻轉
                self.NowScreen = flipped_screen
                # 顯示翻轉後的畫面
                cv2.imshow("Camera", flipped_screen)
        self.ControlWindow.after(10, self.showing_video)  # 以10毫秒間隔進行更新
    
    def take_pic(self):
        # 使用新線程進行圖片生成
        if self.NowScreen is not None:
            threading_thread = Thread(target=self.generate_image)
            threading_thread.start()

    def generate_image(self):
        # Convert the current frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(self.NowScreen, cv2.COLOR_BGR2RGB))
        #image = Image.open(pic_path).convert("RGB")  # 從指定路徑加載圖片
        image = image.resize((512, 512))
        # Display the image
        image.show()

        # Get the strength value from the slider
        strength = float(self.strength_var.get())

        # Generate the AI image
        #result = self.pipe(
        #    prompt=self.prompt_var.get(),
        #    image=image,
        #    strength=strength,
        #    guidance_scale=8.5
        #).images[0]
        result = self.pipe(prompt=self.prompt_var.get(), image=image, strength=strength, guidance_scale=5.5).images[0]

        # Display the result
        result.show()

    




if __name__ == "__main__":
    root = Tk()
    app = Miyazaki_hayao_style_transfer(root)
    root.mainloop()