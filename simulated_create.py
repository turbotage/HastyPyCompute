import simulate_mri as simri

if __name__ == '__main__':
	simri.simulate(dirpath='/home/turbotage/Documents/4DRecon/', imagefile='background_corrected.h5',
			create_crop_image=True, load_crop_image=False,
			create_kdata=True, load_kdata=False,
			nspokes=1000,
			samp_per_spoke=300,#samp_per_spoke=489,
			#method='PCVIPR', # PCVIPR, MidRandom
			method='FullCartesian',
			crop_factor=1.3,
			just_plot=False,
			also_plot=True)