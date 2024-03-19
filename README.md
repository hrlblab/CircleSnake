# CircleSnake: Instance Segmentation with Circle Representation
The official implementation of CircleSnake

[//]: # (![]&#40;docs/fig1.png&#41;)
<img src="docs/fig1.png" alt="drawing" width="600"/>

Contact: [ethan.h.nguyen@vanderbilt.edu](mailto:ethan.h.nguyen@vanderbilt.edu). Feel free to reach out with any questions or discussions!  

## Abstract
In this work, we propose CircleSnake, a simple end-to-end circle contour deformation-based segmentation method for ball-shaped medical objects. Compared to the prevalent DeepSnake method, our contribution is threefold: 

(1) We replace the complicated **bounding box to octagon contour** transformation with a computation-free and consistent **bounding circle to circle contour** adaption for segmenting ball-shaped medical objects; 

(2) Circle representation has **fewer degrees of freedom** (DoF=2) as compared with the octagon representation (DoF=8), thus yielding a more robust segmentation performance and better rotation consistency; 

(3) To the best of our knowledge, the proposed CircleSnake method is the **first end-to-end circle representation deep segmentation pipeline method** with consistent circle detection, circle contour proposal, and circular convolution. 

Glomeruli are used to evaluate the performance of the benchmarks. From the results, CircleSnake increases the average precision of glomerular detection from 0.559 to 0.614. The Dice score increased from 0.804 to 0.849.

## Highlights 

- **Simple:** One-sentence summary: Instead of the conventional bounding box, we propose using a bounding circle to detect ball-shaped biomedical objects.

- **State-of-the-art:** Our CircleSnake method outperforms baseline methods.

- **Fast:** Only requires a single network forward pass.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation instructions.

## Development

Please refer to [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for guidance to train, inferences, and develop our work.

## License

CircleSnake is released under [this](LICENSE) license.

## Citation
~~~
@misc{xiong2024circle,
      title={Circle Representation for Medical Instance Object Segmentation}, 
      author={Juming Xiong and Ethan H. Nguyen and Yilin Liu and Ruining Deng and Regina N Tyree and Hernan Correa and Girish Hiremath and Yaohong Wang and Haichun Yang and Agnes B. Fogo and Yuankai Huo},
      year={2024},
      eprint={2403.11507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{nguyen2022circlesnake,
      title={CircleSnake: Instance Segmentation with Circle Representation}, 
      author={Ethan H. Nguyen and Haichun Yang and Zuhayr Asad and Ruining Deng and Agnes B. Fogo and Yuankai Huo},
      year={2022},
      eprint={2211.01254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
~~~
