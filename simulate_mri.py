import numpy as np
import h5py

import matplotlib.pyplot as plt
import cupy as cp
import cufinufft

from scipy.ndimage import gaussian_filter

import plot_utility as pu
import orthoslicer as ort

import image_creation as ic

import math

import os
import fnmatch
import re

def rename_h5(name, appending):
	return name[:-3] + appending + '.h5'

def crop_image(inpath, outpath, imagefile, just_plot=False, also_plot=False):
	map_in_joiner = lambda path: os.path.join(inpath, path)
	map_out_joiner = lambda path: os.path.join(outpath, path)

	print('Loading images')
	with h5py.File(map_in_joiner(imagefile), "r") as f:
		img = f['corrected_img'][()]
		cd = f['cd'][()]
		smaps = f['smaps'][()]
		
	nlen = 72
	nx = 70
	ny = 70
	nz = 130

	crop_box = [(nx,nx+nlen),(ny,ny+nlen),(nz,nz+nlen)]
	print('Cropping')
	new_img = ic.crop_5d_3d(img, crop_box)
	new_cd = ic.crop_4d_3d(cd, crop_box).astype(np.float32)
	new_smaps = ic.crop_4d_3d(smaps, crop_box)

	new_cd = gaussian_filter(new_cd, sigma=0.8)

	vessel_mask = new_cd > np.quantile(new_cd, 0.995)
	vessel_mask = np.all(vessel_mask, axis=0)

	if just_plot or also_plot:
		ort.image_nd(new_img)
		ort.image_nd(new_cd)
		#ort.image_nd(new_cd + vessel_mask[None,...]*50.0)
		ort.image_nd(vessel_mask.astype(np.float32), max_clim=True)

		if just_plot:
			return (None, None)

	print('Writing cropped images')
	with h5py.File(map_out_joiner(rename_h5(imagefile, '_cropped')), "w") as f:
		f.create_dataset('img', data=new_img)
		f.create_dataset('smaps', data=new_smaps)
		f.create_dataset('cd', data=new_cd)
		f.create_dataset('vessel_mask', data=vessel_mask)

	img = new_img
	smaps = new_smaps
	print('\nCreated cropped images\n')


	return img, smaps


def nufft_of_enced_image(img, smaps, inpath, outpath, 
	nspokes, nsamp_per_spoke, method, crop_factor=1.0, also_plot=False):
	
	map_out_joiner = lambda path: os.path.join(outpath, path)

	frame_coords = []
	frame_kdatas = []

	nframes = img.shape[0]
	nenc = img.shape[1]
	nsmaps = smaps.shape[0]

	im_size = (img.shape[2], img.shape[3], img.shape[4])
	num_voxels = math.prod(list(im_size))

	largest_kdatas = []

	print('Simulating MRI camera')
	for frame in range(nframes):
		print('Frame: ', frame, '/', nframes)

		encode_coords = []
		encode_kdatas = []

		xangles = np.random.rand(nspokes).astype(np.float32)
		xangles = np.arccos(1.0 - 2.0*xangles).astype(np.float32)
		zangles = 2 * np.pi * np.random.rand(nspokes).astype(np.float32)

		coord = np.ascontiguousarray(ic.create_coords(nspokes, nsamp_per_spoke, im_size, method, False, crop_factor, xangles, zangles))

		if also_plot and frame == 0:
			pu.scatter_3d(coord)
			plt.figure()
			plt.plot(xangles, zangles, 'b*')
			plt.show()
			print(' (N K-Space Points) / (N Voxels) ratio: ', coord.shape[1] / num_voxels)				
			print(' (N K-Space Points) / (N Voxels) ratio: ', coord.shape[1] / num_voxels)


		for encode in range(nenc):
			print('Encode: ', encode, '/', nenc, '  Creating coordinates')

			coil_kdatas = []

			coord = cp.array(coord)

			outkdata = cp.empty((coord.shape[1],), dtype=cp.complex64)

			for smap in range(nsmaps):
				print('\r Coil: ', smap, '/', nsmaps, end="")
				coiled_image = cp.array(img[frame,encode,...] * smaps[smap,...])
				cufinufft.nufft3d2(x=coord[0,:], y=coord[1,:], z=coord[2,:], data=coiled_image, 
					out=outkdata)
				outkdata /= np.sqrt(coiled_image.size)

				coil_kdatas.append(outkdata.copy())
			print("")

			encode_coords.append(coord.get())
			encode_kdatas.append(cp.stack(coil_kdatas, axis=0).get())

		largest_kdatas.append(max([np.quantile(np.abs(kd), 0.90).max() for kd in encode_kdatas]))

		frame_coords.append(encode_coords)
		frame_kdatas.append(encode_kdatas)

	largest_kdatas = max(largest_kdatas)

	with h5py.File(map_out_joiner('simulated_coords_kdatas.h5'), "w") as f:
		for i in range(nframes):
			for j in range(nenc):
				ijstr = str(i)+'_e'+str(j)
				f.create_dataset('coords_f'+ijstr, data=frame_coords[i][j])
				f.create_dataset('kdatas_f'+ijstr, data=frame_kdatas[i][j] / largest_kdatas)
		f.create_dataset('maxkdata', data=largest_kdatas)

	print('\nCreated coords and kdatas\n')


	return frame_coords, frame_kdatas

def simulate(
		inpath='', 
		outpath='', 
		imagefile='',
		nspokes=500,
		samp_per_spoke=489,
		method='PCVIPR',
		crop_factor=2.0,
		noise_level=0.0,
		just_plot=False,
		also_plot=False):

	if outpath == '':
		outpath = inpath + f"run_nspoke{nspokes}_samp{samp_per_spoke}_noise{noise_level:.2e}/"

	from pathlib import Path
	Path(outpath).mkdir(parents=True, exist_ok=True)

	img, smaps = crop_image(inpath, outpath, imagefile, just_plot, also_plot)

	if just_plot:
		return

	coords, kdatas = nufft_of_enced_image(img, smaps, inpath, outpath, 
		nspokes, samp_per_spoke, method, crop_factor, also_plot)
	
	return coords, kdatas
	
	

def load_coords_kdatas(dirpath):
	map_joiner = lambda path: os.path.join(dirpath, path)

	def frames_and_encodes(keys):
		framelist = list()
		encodelist = list()
		for key in keys:
			m = re.findall(r'coords_f\d+_e0', key)
			if len(m) != 0:
				framelist.append(m[0])
			m = re.findall(r'coords_f0_e\d+', key)
			if len(m) != 0:
				encodelist.append(m[0])

		return (len(framelist), len(encodelist))
		
	nframes = 0
	nenc = 0
	frame_coords = []
	frame_kdatas = []

	with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "r") as f:
		nframes, nenc = frames_and_encodes(list(f.keys()))
		for i in range(nframes):
			encode_coords = []
			encode_kdatas = []
			for j in range(nenc):
				ijstr = str(i)+'_e'+str(j)
				encode_coords.append(f['coords_f'+ijstr][()])
				encode_kdatas.append(f['kdatas_f'+ijstr][()])
			frame_coords.append(encode_coords)
			frame_kdatas.append(encode_kdatas)
			kdatamax = f['maxkdata'][()]
	print('\nLoaded coords and kdatas\n')
	return (frame_coords, frame_kdatas, nframes, nenc, kdatamax)

def load_smaps(dirpath):
	map_joiner = lambda path: os.path.join(dirpath, path)
	with h5py.File(map_joiner('background_corrected_cropped.h5'), "r") as f:
		smaps = f['smaps'][()]
	return smaps

def load_true_and_smaps(dirpath):
	map_joiner = lambda path: os.path.join(dirpath, path)
	with h5py.File(map_joiner('background_corrected_cropped.h5'), "r") as f:
		true_image = f['img'][()]
		vessel_mask = f['vessel_mask'][()]
		smaps = f['smaps'][()]
	return true_image, vessel_mask, smaps