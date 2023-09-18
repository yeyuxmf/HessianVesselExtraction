# HessianVesselExtraction
Retinal vessel extraction,  opencv_python, skimage, python
1. It also supports tubular object extraction/segmentation.
2. Here, a custom Gaussian kernel is used, followed by convolutional filtering to further obtain the Hessian matrix. Finally, the eigenvalues are calculated and the vascular response is obtained.
3. The second method is to directly obtain the Hessian eigenvalues using the skimage package, thereby obtaining the vascular response.
   
# References
1. Frangi A F, Niessen W J, Vincken K L, et al. Multiscale vessel enhancement filtering International Conference on Medical Image Computing and Computer-Assisted Intervention[C]//Medical Image Computing and Computer Assisted Intervention.
# Environment
1. opencv_python
2. scikit-image  0.19.3
3. python3.7
   
# Experimental 
![vessel](https://github.com/huang229/HessianVesselExtraction/assets/29627190/b07956af-911c-4d28-8458-186d271e9a10)

# License and Citation
1.Without permission, the design concept of this model shall not be used for commercial purposes, profit seeking, etc.

2.The usage rights strictly follow the relevant copyright regulations of the cited paper above. This is only a reproduction.



