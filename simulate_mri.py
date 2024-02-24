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

def crop_image(dirpath, imagefile, create_crop_image=False, load_crop_image=False, just_plot=False, also_plot=False):
	map_joiner = lambda path: os.path.join(dirpath, path)

	if create_crop_image and load_crop_image:
		raise RuntimeError('Can not both load and create crop images')

	if create_crop_image:
		print('Loading images')
		with h5py.File(map_joiner(imagefile), "r") as f:
			img = f['corrected_img'][()]
			cd = f['cd'][()]
			smaps = f['smaps'][()]
			
		nlen = 72
		nx = 70
		ny = 70
		nz = 130 #45

		#nx = 60
		#ny = 60
		#nz = 70

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
		with h5py.File(map_joiner(rename_h5(imagefile, '_cropped')), "w") as f:
			f.create_dataset('img', data=new_img)
			f.create_dataset('smaps', data=new_smaps)
			f.create_dataset('cd', data=new_cd)
			f.create_dataset('vessel_mask', data=vessel_mask)

		img = new_img
		smaps = new_smaps
		print('\nCreated cropped images\n')
	elif load_crop_image:
		print('Loading cropped images')
		with h5py.File(map_joiner(rename_h5(imagefile, '_cropped')), "r") as f:
			img = f['img'][()]
			smaps = f['smaps'][()]
		print('\nLoaded cropped images\n')

	return img, smaps


def nufft_of_enced_image(img, smaps, dirpath, 
	nspokes, nsamp_per_spoke, method, crop_factor=1.0,
	create_kdata=False, load_kdata=False, also_plot=False):
	
	nfreq = nspokes * nsamp_per_spoke
	map_joiner = lambda path: os.path.join(dirpath, path)

	if create_kdata and load_kdata:
		raise RuntimeError('Can not both load and create nufft_of_enced_image images')

	frame_coords = []
	frame_kdatas = []

	if create_kdata:
		nframes = img.shape[0]
		nenc = img.shape[1]
		nsmaps = smaps.shape[0]

		#coords = np.empty((nframes,nenc,3,nfreq), dtype=np.float32)
		#kdatas = np.empty((nframes,nenc,nsmaps,nfreq), dtype=np.complex64)

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

			#xangles = np.pi * np.linspace(0, 1, nspokes, endpoint=False).astype(np.float32)
			#zangles = 2*np.pi * np.ones((nspokes,)).astype(np.float32) #np.ones(0, 1, nspokes, endpoint=False).astype(np.float32)

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

		with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "w") as f:
			for i in range(nframes):
				for j in range(nenc):
					ijstr = str(i)+'_e'+str(j)
					f.create_dataset('coords_f'+ijstr, data=frame_coords[i][j])
					f.create_dataset('kdatas_f'+ijstr, data=frame_kdatas[i][j] / largest_kdatas)
			f.create_dataset('maxkdata', data=largest_kdatas)

		print('\nCreated coords and kdatas\n')
	elif load_kdata:
		frame_coords = np.array([], dtype=object)
		frame_kdatas = np.array([], dtype=object)
		with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "r") as f:
			for i in range(nframes):
				encode_coords = np.array([], dtype=object)
				encode_kdatas = np.array([], dtype=object)
				for j in range(nenc):
					ijstr = str(i)+'_e'+str(j)
					np.append(encode_coords, f['coords_f'+ijstr])
					np.append(encode_kdatas, f['kdatas_f'+ijstr])
		print('\nLoaded coords and kdatas\n')

	return frame_coords, frame_kdatas

def simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='images_6f.h5',
		create_crop_image=False, load_crop_image=False, 
		create_kdata=False, load_kdata=False,
		nimgout=20,
		nspokes=500,
		samp_per_spoke=489,
		method='PCVIPR',
		crop_factor=2.0,
		just_plot=False,
		also_plot=False):

	img, smaps = crop_image(dirpath, imagefile, create_crop_image, load_crop_image, just_plot, also_plot)

	if just_plot:
		return #ort.image_nd(img)

	coords, kdatas = nufft_of_enced_image(img, smaps, dirpath, 
		nspokes, samp_per_spoke, method, crop_factor,
		create_kdata, load_kdata, also_plot)
	
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