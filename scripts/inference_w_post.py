import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from training.coach import Coach

def run():
	test_opts = TestOptions().parse()

	if test_opts.resize_factors is not None:
		assert len(test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
		                                'downsampling_{}'.format(test_opts.resize_factors))
	else:
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	os.makedirs(out_path_results, exist_ok=True)

	out_path_coupled = None
	if test_opts.couple_outputs:
		if test_opts.resize_factors is not None:
			out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
			                                'downsampling_{}'.format(test_opts.resize_factors))
		else:
			out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
		os.makedirs(out_path_coupled, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
	                           transform=transforms_dict['transform_inference'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)
	train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
		                                     target_root=dataset_args['train_target_root'],
		                                     source_transform=transforms_dict['transform_source'],
		                                     target_transform=transforms_dict['transform_gt_train'],
		                                     opts=opts)
	train_dataloader = DataLoader(train_dataset,
								   batch_size=1,
								   shuffle=False,
								   num_workers=int(0),
								   drop_last=False)

	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	
	print("Inverting Training Image")
	coach = Coach(opts)
	coach.invert_training_img(overwrite=False)
		
	print("Computing Feature Vectors")
	x_list, codes_list = [], []
	loss_mask = torch.zeros(opts.label_nc).cuda()
	for input_batch in train_dataloader:
		x, _, file_names = input_batch
		x = x.cuda().float()
		for bid in range(x.shape[0]):
			loss_mask[torch.argmax(x,dim=1).unique()]=1.
			x_list.append(x[bid].unsqueeze(0))
			codes_list.append(torch.load(os.path.join(os.path.dirname(dataset_args['train_target_root'])+'/inversion',os.path.splitext(file_names[bid])[0]+'.pt'))['codes'])

	with torch.no_grad():
		net.compute_rep_vec(x_list, codes_list)
				
	print("Computing PCA")
	if not os.path.exists(os.path.dirname(test_opts.checkpoint_path)+'/pca.pt'):
		vecs = []
		N = 1024
		with torch.no_grad():
			for i in range(N):
				vec_to_inject = torch.randn(1, 512).cuda()
				_, latent_to_inject = net(vec_to_inject,
				                          input_code=True,
				                          return_latents=True)
				vecs.append(latent_to_inject)
			
		vecs = torch.cat(vecs)
		b, n, c = vecs.shape
		mu = vecs.mean(dim=0).unsqueeze(0)
		dev = vecs - mu.repeat((N,1,1))
		cov = torch.bmm(dev.permute(1,2,0), dev.permute(1,0,2))/N
		eigenvalues, eigenvectors = torch.symeig(cov, eigenvectors=True)
		stats_dict = {'mu':mu, 'eigenvalues':eigenvalues, 'cov':cov, 'eigenvectors':eigenvectors}
		torch.save(stats_dict, os.path.dirname(test_opts.checkpoint_path)+'/pca.pt')
		
	PCA_dict = torch.load(os.path.dirname(test_opts.checkpoint_path)+'/pca.pt')
	mu = PCA_dict['mu']
	eigenvectors = PCA_dict['eigenvectors']
	
	print("Optimization")
	global_i = 0
	max_iter = 201
	save_interval = 10
	st_dim = 504
	ed_dim = 512
	L = 8
	mask_classes = []
	if opts.sparse_labeling:
		mask_classes.append(0)
		
	global_time = []
	for input_batch in tqdm(dataloader):
		input_batch, _ = input_batch
		if global_i > opts.n_images:
			break
		with torch.no_grad():
			input_cuda = input_batch.cuda().float()
			tic = time.time()
			result_batch, codes_batch = run_on_batch(input_cuda, net, opts)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			im_path = dataset.paths[global_i]
			codes = codes_batch[i]
			initial_codes = codes.clone().detach()
			

			g_partial = torch.bmm(torch.inverse(eigenvectors), (codes-mu).permute(1,2,0))
			optimized_codes = g_partial[:1,st_dim:ed_dim].clone()
			optimized_codes = torch.nn.Parameter(optimized_codes.data)	
			optimizer = Adam([optimized_codes], lr=0.05)
			C = net.id2prototype.unsqueeze(0)
			
			for iter in range(max_iter):
				g = torch.cat([g_partial[:1,:st_dim], optimized_codes, g_partial[:1,ed_dim:]], dim=1).repeat((opts.style_num,1,1))
				codes = torch.bmm(eigenvectors, g).permute(2,0,1) + mu
				codes[:,L:] = initial_codes[:,L:]
				y_hat, latent, feature_map = net.decoder([codes],
			                                     input_is_latent=True,
			                                     randomize_noise=False,
			                                     return_latents=True,
			                                     return_feature_map=True)
				y_hat = net.face_pool(y_hat)
				result = tensor2im(y_hat[0])
				loss = optimization_loss(feature_map, input_cuda, C, opts, mask_classes)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				if iter % save_interval == 0:
					print(loss)
					if opts.couple_outputs or global_i % 100 == 0:
						input_resized = log_input_image(input_batch[i], opts)
						if opts.resize_factors is not None:
							# for super resolution, save the original, down-sampled, and output
							source = Image.open(im_path)
							res = np.concatenate([np.array(source.resize((256, 256))),
							                      np.array(input_resized.resize((256, 256), resample=Image.NEAREST)),
							                      np.array(result.resize((256, 256)))], axis=1)
						else:
							# otherwise, save the original and output
							res = np.concatenate([np.array(input_resized.resize((256, 256))),
							                      np.array(result.resize((256, 256)))], axis=1)
		
						Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.splitext(os.path.basename(im_path))[0]+'_'+str(iter)+'.png'))
		
					im_save_path = os.path.join(out_path_results, os.path.splitext(os.path.basename(im_path))[0]+'_'+str(iter)+'.png')
					Image.fromarray(np.array(result.resize((256, 256)))).save(im_save_path)

			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, net, opts):
	result_batch = []
	codes_batch = []
	if opts.latent_mask is None:
		result_batch, codes = net(inputs, randomize_noise=False, return_latents=True)
		codes_batch.append(codes)
	else:
		latent_mask = [int(l) for l in opts.latent_mask.split(",")]
		for image_idx, input_image in enumerate(inputs):
			# get latent vector to inject into our input image
			vec_to_inject = np.random.randn(1, 512).astype('float32')
			_, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
			                          input_code=True,
			                          return_latents=True)

			# get output image with injected style vector
			res, codes = net(input_image.unsqueeze(0).to("cuda").float(),
			          latent_mask=latent_mask,
			          inject_latent=latent_to_inject,
			          alpha=opts.mix_alpha, 
			          return_latents=True)
			codes_batch.append(codes)
			result_batch.append(res)
		result_batch = torch.cat(result_batch, dim=0)
	
	return result_batch, codes_batch

def optimization_loss(feature_map, x, C, opts, mask_classes=[]):
	b, c, h, w = feature_map.shape
	hw = h*w
	feature_map_reshaped = feature_map.reshape(b,c,hw)
	x_reshaped = F.interpolate(x.float(), size=(h,w)).reshape(b,opts.label_nc,hw).permute(0,2,1)
	xC = torch.bmm(x_reshaped, C).permute(0,2,1)
	x_reshaped_normalized = F.normalize(x_reshaped, p=1, dim=1)
	xF = torch.bmm(x_reshaped_normalized.permute(0,2,1), feature_map_reshaped.permute(0,2,1))

	mask = torch.ones_like(feature_map_reshaped)
	for cid in mask_classes:
		mask.permute(0,2,1)[x_reshaped[:,:,cid]==1] = 0.
		
	loss = 1.-F.cosine_similarity(mask*feature_map_reshaped, mask*xC, dim=1).mean()

	return loss

if __name__ == '__main__':
	run()
