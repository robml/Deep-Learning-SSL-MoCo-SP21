## Deep Learning Spring 2021 Competition
## An Adaptation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a PyTorch adaptation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```

### Background
This team composed of **Robert Melikyan, Jash Tejaskumar Doshi, and Omkar Ajit Darekar**; was tasked with creating a self-supervised deep learning model for Spring 2021. Given the impressive results by MoCo and a few initial tests between comparative models like SIMCLR, BOYL, or SWAV, the team settled on MoCo as the basis of its architecture to satisfy the tradeoff between code dependencies, computation intensity and model effectiveness.

### Preparation

Ensure PyTorch is installed along with 96x96 dataset of RGB images. Although given an original dataset of both labelled and unlabelled images, 12800 images were labelled during the process. Their respective identification filenames and labels can be found [here](https://drive.google.com/drive/folders/1SxcXDGZpbkNeIScJA3dFK1aMcimK8XxF?usp=sharing).

The two primary files for this task can be found below, in all the code samples below, the files are maintained and executed in the same directory:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on the dataset in an 2-gpu machine, run a sbatch file as specified below:
```

```

### HyperParemeters

Initially, this script uses all the default hyper-parameters as described in the MoCo v2 paper.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 2-gpu machine, run the following sbatch request:
```
#!/bin/bash                                                                     

#SBATCH --gres=gpu:2                                                            
#SBATCH --partition=n1s16-t4-2                                                  
#SBATCH --account=dl17                                                          
#SBATCH --time=20:00:00                                                         
#SBATCH --output=moco%j.out                                                     
#SBATCH --error=moco%j.err                                                      
#SBATCH --exclusive                                                             
#SBATCH --requeue                                                               

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/test/repo/NYU_DL_comp/moco/

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "                
source /ext3/env.sh                                                             
conda activate dev                                                              
CUDA_VISIBLE_DEVICES=0,1                                                        
python3 main_linclssupervised.py \                                              
  -a resnet50 \                                                                 
 -data /dataset \                                                               
  --lr 30.0 \                                                                   
  --batch-size 256 \                                                            
  --pretrained $SCRATCH/checkpoints/demo/moco_unsupervised_0065.pth.tar \       
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size\
 1 --rank 0"
```
# 100 epochs

Linear classification results on CSCI-GA.2572 dataset using this repo with 2 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">44 hours</td>
<td align="center">33&plusmn;0.1</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std: the 5 results of MoCo v2 are {67.7, 67.6, 67.4, 67.6, 67.3}.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo v1</a></td>
<td align="center">200</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">60.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>b251726a</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>a04e12f8</tt></td>
</tr>
</tbody></table>

REQUIRED PACKAGES
-------------------
- Python 3 (tested on 3.8)
- PyTorch (1.8.1)
- CudaToolKit (tested on 11.1)
- SciPy
- Numpy

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.