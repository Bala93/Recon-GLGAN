# ReconGLGAN - GANs for MRI Reconstruction

This code is based on the [fastMRI code](https://github.com/facebookresearch/fastMRI) from Facebook Research.

## Dataset

TBA


## Training 

Set the appropriate paths to training and test data in ```train.sh```. Then,

```sh train.sh```


## Validation

Set the appropriate path to the model checkpoint in ```valid.sh```. Then,

```sh train.sh```.


## Evaluation

```python common/patch_eval_onecoord.py --target-path [PATH_TO_TARGET_DATA] --predictions-path [PATH_TO_PREDICTED_OUTPUTS] --acceleration [] --roi-size [SIZE_OF_ROI]```