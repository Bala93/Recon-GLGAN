sudo /home/htic/anaconda2/envs/torch4/bin/python -W ignore models/unetGAN/run_unetGAN.py --checkpoint /media/htic/NewVolume5/miccai_mri/models_2x/globaldisc_dagan/best_ssim_model.pt --out-dir /media/htic/NewVolume5/miccai_mri/reconstructions_2x/globaldisc_dagan/best_ssim/
sudo /home/htic/anaconda2/envs/torch4/bin/python -W ignore models/unetGAN/run_unetGAN.py --checkpoint /media/htic/NewVolume5/miccai_mri/models_2x/globaldisc_dagan/best_mse_ssim_model.pt --out-dir /media/htic/NewVolume5/miccai_mri/reconstructions_2x/globaldisc_dagan/best_mse_ssim/
sudo /home/htic/anaconda2/envs/torch4/bin/python -W ignore models/unetGAN/run_unetGAN.py --checkpoint /media/htic/NewVolume5/miccai_mri/models_2x/globaldisc_dagan/best_mse_model.pt --out-dir /media/htic/NewVolume5/miccai_mri/reconstructions_2x/globaldisc_dagan/best_mse/