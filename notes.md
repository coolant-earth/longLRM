
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
apt-get update && apt-get install -y unzip
unzip awscliv2.zip
./aws/install
```
```
torchrun --standalone --nproc_per_node=8 main.py --config configs/stage07kGT.yaml --default-config configs/dl3dv_i256_32input_8target.yaml
```


12:01 PM training on just 18 scenes (2 batches )

1. problem observed - gaussians in the back can be avoided complelety



