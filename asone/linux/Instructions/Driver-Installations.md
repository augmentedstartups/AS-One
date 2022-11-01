# Driver Installations


### Linux

For systems with `GPU` please verify you have nvidia drivers installed.

Run

```
nvidia-smi
```
Drivers are installed if you see following.

![](../imgs/nvidia-drivers.png)

If drivers are not installed, you can do so using following command:

```
sudo apt-get install nvidia-driver-YYY nvidia-dkms-YYY
```
where,
- `YYY`= Nvidia driver version

e.g `sudo apt-get install nvidia-driver-510 nvidia-dkms-510`

- `Reboot` your system after installing nvidia-drivers.
```
sudo reboot
```


Return to [Installation Page](../../../README.md) 
