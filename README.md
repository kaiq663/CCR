# Consensual Contrastive Regularization Acting as Preserving Force for Self-Supervised Generalization of Real Image Denoising


## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [NAFNet](https://github.com/megvii-research/NAFNet) 


## Dataset

Training dataset : [SIDD](https://abdokamel.github.io/sidd/#sidd-medium)
Evaluation datasets : [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset), [CC](https://github.com/csjunxu/MCWNNM-ICCV2017), HighISO, iPhone, Huawei.
Additioanl real-world noise datasets can be downloaded from "https://github.com/ZhaomingKong/Denoising-Comparison"<br><br>


## QuickStart

For test
python3 -m torch.distributed.launch --nproc_per_node=1 basicsr/test.py -opt options/test/DnCNN.yml -name=AFM_test --launcher pytorch



## Results and Pre-trained model

| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSRN| 37.75  | 36.84 | 39.17   | 40.65   | 38.39   | 38.56   |
|SSIM| 0.9804 | 0.9830 | 0.9801  | 0.9777  | 0.9683  | 0.9779  |
