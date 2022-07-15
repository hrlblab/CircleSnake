# CircleSnake: Instance Segmentation with Circle Representation
The official implementation of CircleSnake

![](docs/fig1.png)

Contact: [ethan.h.nguyen@vanderbilt.edu](mailto:ethan.h.nguyen@vanderbilt.edu). Feel free to reach out with any questions or discussion!  

## Abstract
Circle representation has recently been introduced as a "medical imaging optimized" representation for more effective instance object detection on ball-shaped medical objects. With its superior performance on instance detection, it is appealing to extend the circle representation to instance medical object segmentation. In this work, we propose CircleSnake, a simple end-to-end circle contour deformation-based segmentation method for ball-shaped medical objects. Compared to the prevalent DeepSnake method, our contribution is threefold: 

(1) We replace the complicated **bounding box to octagon contour** transformation with a computation-free and consistent **bounding circle to circle contour** adaption for segmenting ball-shaped medical objects; 

(2) Circle representation has **fewer degrees of freedom** (DoF=2) as compared with the octagon representation (DoF=8), thus yielding a more robust segmentation performance and better rotation consistency; 

(3) To the best of our knowledge, the proposed CircleSnake method is the **first end-to-end circle representation deep segmentation pipeline method** with consistent circle detection, circle contour proposal, and circular convolution. The key innovation is to integrate the circular graph convolution with circle detection into an end-to-end instance segmentation framework, enabled by the proposed simple and consistent circle contour representation. Glomeruli are used to evaluate the performance of the benchmarks. 

From the results, CircleSnake increases the average precision of glomerular detection from 0.559 to 0.614. The Dice score increased from 0.804 to 0.849.

## Highlights 

- **Simple:** One-sentence summary: Instead of the conventional bounding box, we propose using a bounding circle to detect ball-shaped biomedical objects.

- **State-of-the-art:** Our CircleSnake method outperforms baseline methods.

- **Fast:** Only requires a single network forward pass.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation instructions.

## License

CircleSnake itself is released under [this](docs/LICENSE) license.
