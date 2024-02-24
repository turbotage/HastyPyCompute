import simulate_mri as simri

if __name__ == '__main__':
	simri.simulate(dirpath='/home/turbotage/Documents/4DRecon/', imagefile='background_corrected.h5',
			create_crop_image=True, load_crop_image=False,
			create_kdata=True, load_kdata=False,
			nspokes=250,
			samp_per_spoke=85,#samp_per_spoke=489,
			#method='PCVIPR', # PCVIPR, MidRandom
			method='PCVIPR',
			crop_factor=1.4,
			just_plot=False,
			also_plot=True)