## Tomo_Processing_Pipeline Usage
There are several defined functions in this package and it follows the Tomo processing pipeline. You can use each of the function seperately. 
The functions include Motion Correction, Image Stacking, Image Alignment, Image CTF Calculation, Image Reconstruction, Image Particle Picking, Coordinate Preparation, Relion Star Preparation, Star File Splitting and Image Averaging. (Noted there are several third party packages needed but you can download them easily from their website).
Below give some examples:
1. Run Motioncorrection command to deal a set of images
```
python3 Tomo2SPA.py motioncor foldername gain [--swap_xy] [--output_star] [--image_extension] [--frame_number] [--pixel_size] [--patch] [--kV] [--Ftbin] [--Gpu] [--GpuMemUsage] 
``` 
2. Run Image Stack command to deal a set of images
```
python3 Tomo2SPA.py stack foldername [--swap_xy] [--output_star] [--frame_number]
```
This code belongs to Dr.Hao Xiong from Yale University. If you have any question, please don't hesitate to contact hao.xiong@yale.edu

<!---
xiong19912010/xiong19912010 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

Image particle picking after using functions (particle_picking)\n


<img width="319" alt="image" src="https://user-images.githubusercontent.com/94659159/179382531-6c285e60-e64a-40b5-af0d-77429e04f2bf.png">

