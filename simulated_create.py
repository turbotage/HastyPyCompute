import simulate_mri as simri

if __name__ == '__main__':
	simri.simulate(
			inpath='/home/turbotage/Documents/4DRecon/', 
			outpath='',
			imagefile='background_corrected.h5',
			nspokes=200,
			samp_per_spoke=100,
			method='PCVIPR',
			crop_factor=1.4,
			noise_level=0.0002,
			just_plot=False,
			also_plot=True)