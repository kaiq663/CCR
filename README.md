# Consensual Contrastive Regularization Acting as Preserving Force for Self-Supervised Generalization of Real Image Denoising


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
Validate on SIDD Validation dataset,
```
cd validate
python validate_SIDD.py
```




## Results and Pre-trained model

| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSRN| 37.81  | 36.61 | 39.43   | 41.10   | 38.89   | 38.77   |
|SSIM| 0.9587 | 0.9494 | 0.9666  | 0.9711  | 0.9588  | 0.9609  |
