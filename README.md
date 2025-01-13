# Roubst Self-Supervised Real Image Denoising via Consensual Contrastive Regularization as Preserving Force


## Dataset

Training dataset : [SIDD](https://abdokamel.github.io/sidd/#sidd-medium)
Evaluation datasets : [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset), [CC](https://github.com/csjunxu/MCWNNM-ICCV2017), HighISO, iPhone, Huawei.
Additioanl real-world noise datasets can be downloaded from "https://github.com/ZhaomingKong/Denoising-Comparison"<br><br>

### Training 
Training on SIDD Medium dataset,
```
sh train.sh
```

### Validation
Validate on SIDD validation dataset , and all OOD datasets,
```
cd validate
python validate_SIDD.py
```




## Results and Pre-trained model
PUCA officially pre-trained
| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSNR| 37.51  | 36.47 | 39.41   | 41.04   | 37.80   | 38.45   |

only pushing force
| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSNR| 37.74  | 36.59 | 39.19   | 40.83   | 39.00   | 38.67   |

pushing  & preserving force
| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSNR| 37.81  | 36.61 | 39.43   | 41.10   | 38.89   | 38.77   |
|SSIM| 0.9587 | 0.9494 | 0.9666  | 0.9711  | 0.9588  | 0.9609  |
