2023/9/6
# Step1: Prepare the dataset
`python prepare_data.py --size 256 --out celeba_256 E:\01-Research\03-Projects\20230801-暑研项目-Diffusion\02-代码\diffusion_distiller_autoweidata\celeba_256_img`

# Step2: Train the teacher model
`python ./train.py --module celeba_u --name celeba --dname original --batch_size 1 --num_workers 0 --num_iters 150000`
## Sampling
`python ./sample.py --out_file ./images/celeba/original_full/celeba_original_0001.png --module celeba_u --time_scale 1 --checkpoint ./checkpoints/celeba/original/checkpoint.pt --batch_size 1`
### Cropping
`python crop_generated_images.py` (revise the root)
### Evaluation
'pip install pytorch-fid'
'python -m pytorch_fid /dis_dif/images/base_0/full /dis_dif/data/fid_cal --device cuda:0'

# Step3: Distillate the student models progressively

# Step4: Evaluate the models and reweight the training material

## Subgoal 1: Evaluate the models with FID and IS
### FID for diffusion
Firstly, I think about a question: will the class label in CelebA influence the training result of ddpm in this case? The answer is no, after looking up the internet and the code.

Then, we will calculate the FID score of the eight models. First of all, the FID score can represente the distance between two Gaussian distributions, which are real images and sampling images in our case.

1. Get a few images from the training set (at least **5000**)
2. Lightweight preprocessing on them