import sys
#sys.path.insert(0, '../StrainPycon') # force the repo version
import strainpycon
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import trange,tqdm



s = strainpycon.StrainRecon()
s.nopt = 1000 # you may want to increase this if m and/or n increases

N = 10 # number of random samples
m = 24 # number of SNP sites (i.e., length of the measurement vector)
#gamma = 0.05 # standard deviation of noise (in the measurement and in the misfit)
gamma = 0.01
#gamma = 0.001

# Number of strains
nt = [ 1, 2, 3, 4, 5 ]

SKIP_LOWFREQ = True # Dismiss weight vectors with less than 1%
#SKIP_LOWFREQ = False # Dismiss weight vectors with less than 1%


def one_sample(args):
	ntrue,idx = args
	strainmat = np.random.randint(0, 2, (m, ntrue)) # random binary matrix
	while True:
		freqvec = strainpycon.psimplex.rand_simplex(ntrue) # random freq vector
		if not SKIP_LOWFREQ or min(freqvec) >= 0.01:
			break
	freqvec_opt = 2**np.arange(ntrue-1, -1, -1) / (2**ntrue-1)

	# measurement (with noise) and misfits for n=1, ..., ntrue+1
	meas = np.dot(strainmat, freqvec) + gamma*np.random.normal(size=m)
	#mfits = s.misfits(meas, ntrue+1, gamma=gamma)
	mfits = s.misfits(meas, max(nt)+1, gamma=gamma)

	# measurements and misfits if optimal freq vector was used instead
	meas_opt = np.dot(strainmat, freqvec_opt) + gamma*np.random.normal(size=m)
	#mfits_opt = s.misfits(meas_opt, ntrue+1, gamma=gamma)
	mfits_opt = s.misfits(meas_opt, max(nt)+1, gamma=gamma)
	#mfits_opt *= 2

	return (ntrue,idx,mfits,mfits_opt)

def sampler():
	for ntrue in nt:
		for n in range(N):	
			yield (ntrue,n)
mfits = {}
mfits_opt = {}
for ntrue in nt:
	mfits[ntrue] = np.empty((N,max(nt)+1)) #ntrue+1))
	mfits_opt[ntrue] = np.empty((N,max(nt)+1)) #ntrue+1))


with multiprocessing.Pool() as pool:
	with tqdm(total=len(nt)*N) as pbar:
		for i,(ntrue,idx,mf,mf_opt) in tqdm(enumerate(pool.imap_unordered(one_sample, sampler()))):
			pbar.update()
			mfits[ntrue][idx] = mf
			mfits_opt[ntrue][idx] = mf_opt


# frequency vector that maximizes the bi-independency
morozov = m / 2 # right hand side of the discrepancy principle
thresholds = np.logspace(np.log10(morozov/50), np.log10(morozov*50), 400)

correct_probs = {}
correct_probs_opt = {}
for ntrue in nt:
	# count correct MOIs
	correct_probs[ntrue] = [100.0*np.sum(np.sum(mfits[ntrue] > threshold, axis=-1) == ntrue-1)/N for threshold in thresholds]
	correct_probs_opt[ntrue] = [100.0*np.sum(np.sum(mfits_opt[ntrue] > threshold, axis=-1) == ntrue-1)/N for threshold in thresholds]

	#plt.semilogx(thresholds, correct_probs, '-o')
	#plt.semilogx(thresholds, correct_probs_opt, '-o')

def do_plot(name, prbs):
	for ntrue in nt:
		plt.semilogx(thresholds, prbs[ntrue], '-o', label='$n=%d$ strains' % ntrue)

	plt.axvline(x=morozov, color='k', label='Morozov')
	plt.xlabel('Misfit Threshold')
	plt.ylabel('% Correct MOI')
	#plt.title('m={}, n={}, gamma={}'.format(m, ntrue, gamma))
	#plt.legend(['optimal freq', 'random freq', 'morozov'])
	plt.legend(loc='lower right') #['Random ratio', 'Morozov'])

	plt.savefig('figures/moi-%s-gamma=%2.2f.pdf' % (name,gamma))
	plt.show()


do_plot('normal-m={}'.format(m), correct_probs)
do_plot('optfreq-m={}'.format(m), correct_probs_opt)

