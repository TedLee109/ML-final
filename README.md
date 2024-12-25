## Test Image to Image 
To test image to image pipeline, you can run `run_img2img.ipynb`. The first block is to transfer a single image to Miyasaki Hayao's style, and the second block is to tranfer images in a folder at once. 

1. Update paths for input images and LoRA checkpoint.

   You can change `"step_final_lora.safetensors"` to the `.safetensors` file you have trained. Update the following line in the code:
   ```python
   patch_pipe(pipe, "step_final_lora.safetensors", patch_text=True, patch_unet=True, patch_ti=True)
   ```
   Replace `"step_final_lora.safetensors"` with the path to your trained `.safetensors` file.
2. In 1st block, replace `image_path` with the image you want to transfer. In 2nd block, replace `input_folder` with the folder contains the images you want to transfer style, and replace `output_folder` with the folder that you want to store the outputs. 
   
3. Run the cells to generate styled images.

## Train 

1. Prepare your dataset
2. In the `multivector_example.sh` script, change `INSTANCE_DIR` to your dataset path and `OUTPUT_DIR` to the path where you want to store your `.safetensors` file. Update the following lines in the script:
   ```python
   INSTANCE_DIR=/path/to/your/dataset
   OUTPUT_DIR=/path/to/store/safetensors
   ```
3. (Optional) Adjust the parameters in `./training_scripts/multivector_example.sh`
4. Run `multivector_example.sh`
## Evaluation 

All evaluation code and data are under `./evaluation`. To calculate content and style loss for Miyasaki Hayao's style images, run `content_loss.py` and `style_loss.py`.

Before modifying anything, you can already run these two scripts for the Miyasaki Hayao style images we (the authors) generated.

To evaluate your own results:
1. Copy your folder of original images to `./evaluation/origin/`
2. Copy your folder of generated images to `./evaluation/our_model`
3. Copy the folder of target style images to `./evaluation/`
4. Ensure that the images in the original image folder and the generated image folder are in the same order.
5. Update `content_loss.py`:
   - `content_image_folder` with the path to your folder of original images
   - `folder` with the path to your folder of generated images
6. Update `style_loss.py`:
   - `style_image_folder` with the path to your folder of style images
   - `generated_folders` with the path to your folder of generated images
7. Run `style_loss.py` and `content_loss.py`

**Reminder:** Before running the evaluation code, `cd` into the `./evaluation` directory.

## Application

To enable more intuitive usage of our trained model, we developed an application.You can run `app.py` to try style transfer on your selfie photo.

To utilize this application:
1. Install dependencies:
   Ensure you have all necessary libraries installed. You can do this by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your environment:
   Make sure your environment supports GPU acceleration.
3. Run the application:
   Execute the `app.py` file to start the application:
4. Select the desired camera from the dropdown menu and click `Start`. (Usually 0)
5. Click the `Take photo` button to capture a photo, and the application will apply the style transfer and display the result.
6. Click `Close` to exit the application.
