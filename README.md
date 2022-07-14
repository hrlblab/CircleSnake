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

[comment]: <> (## Benchmark Evaluation and Training)

[comment]: <> (After [installation]&#40;docs/INSTALL.md&#41;, follow the instructions in [DATA.md]&#40;docs/DATA.md&#41; to setup the datasets. Then check [GETTING_STARTED.md]&#40;docs/GETTING_STARTED.md&#41; to reproduce the results in the paper.)

[comment]: <> (We provide scripts for all the experiments in the [experiments]&#40;experiments&#41; folder.)

[comment]: <> (## Develop)

[comment]: <> (If you are interested in training CircleNet in a new dataset, use CircleNet in a new task, or use a new network architecture for CircleNet please refer to [DEVELOP.md]&#40;docs/DEVELOP.md&#41;. Also feel free to send us emails for discussions or suggestions.)

## License

CircleSnake itself is released under the MIT License (refer to the LICENSE file for details).
Parts of code and documentation are borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet) and [DeepSnake](https://github.com/zju3dv/snake).

[comment]: <> (## Citation)

[comment]: <> (If you find this project useful for your research, please use the following BibTeX entry.)

[comment]: <> (    @article{nguyen2021circle,)

[comment]: <> (      title={Circle Representation for Medical Object Detection},)

[comment]: <> (      author={Nguyen, Ethan H and Yang, Haichun and Deng, Ruining and Lu, Yuzhe and Zhu, Zheyu and Roland, Joseph T and Lu, Le and Landman, Bennett A and Fogo, Agnes B and Huo, Yuankai},)

[comment]: <> (      journal={IEEE Transactions on Medical Imaging},)

[comment]: <> (      year={2021},)

[comment]: <> (      publisher={IEEE})

[comment]: <> (    })
