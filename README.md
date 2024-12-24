## Test Image to Image 
To test image to image pipeline, you can run `run_img2img.ipynb`. The first block is to transfer a single image to Miyasaki Hayao's style, and the second block is to tranfer images in a folder at once. 

1. Update paths for input images and LoRA checkpoint.

   You can change `"step_final_lora.safetensors"` to the `.safetensors` file you have trained. Update the following line in the code:
   ```
   patch_pipe(pipe, "step_final_lora.safetensors", patch_text=True, patch_unet=True, patch_ti=True)
   ```
   Replace `"step_final_lora.safetensors"` with the path to your trained `.safetensors` file.
2. In 1st block, replace `image_path` with the image you want to transfer. In 2nd block, replace `input_folder` with the folder contains the images you want to transfer style, and replace `output_folder` with the folder that you want to store the outputs. 
   
3. Run the cells to generate styled images.

## Train 

1. Prepare your dataset
2. In the `multivector_example.sh` script, change `INSTANCE_DIR` to your dataset path and `OUTPUT_DIR` to the path where you want to store your `.safetensors` file. Update the following lines in the script:
   ```
   INSTANCE_DIR=/path/to/your/dataset
   OUTPUT_DIR=/path/to/store/safetensors
   ```
4. (Optional) Adjust the parameters in `./training_scripts/multivector_example.sh`
5. run `multivector_example.sh`

## Evaluation 

