curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
apt-get update && apt-get install -y unzip
unzip awscliv2.zip
./aws/install

```
torchrun --standalone --nproc_per_node=1 main.py --config configs/7m1t_tm.yaml --default-config configs/dl3dv_i256_32input_8target.yaml
```


