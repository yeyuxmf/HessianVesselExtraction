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



