import sys
import os
import h5py
import scipy.io as sio
import scipy.stats as sts
import numpy as np
import math
import re
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fnmatch
import csv
import logging as log
import multiprocessing 
from joblib import Parallel,delayed,Memory
from tqdm import tqdm,trange
from strainpycon import StrainRecon


def infertruth(inp):
	truth = { 'A' : [ "000010111111110100011011", "101111011000110111101011", "000111100010000010000000" ],
	          'B' : [ "101111011000110111101011", "100101011111111100111111", "000010000010000001001000" ],
	          'C' : [ "101010111010110011110110", "000111100010000010000000", "100101011111111100111111" ],
	          'C2': [ "000111100010000010000000", "101010111010110011110110", "100101011111111100111111" ]
	        }
	tvec  = { '1' : [ 0.975, 0.02, 0.005 ],
		  '2' : [ 0.95,  0.04, 0.01  ],
		  '3' : [ 0.88,  0.1,  0.02  ] 
		}
	# Formatting here can differ
	out = re.match (r'Sample_([ABC])([1-5])', inp)
	assert out is not None
	exp = str(out.groups()[0])
	mix = str(out.groups()[1])

	# Special case: fix DBS/BLO mutation in A
	if exp == 'A' and any([x in inp for x in ['ZHO4076','ZHO5553','ZHO5019']]):
		# Mutation in base pair 23, add mutated strain from first strain to truth
		new = list(truth[exp][0])
		new[23-1] = '0' if new[23-1] == 1 else '1'
		truth[exp].insert (1, "".join(new)) #make second most prominent
		# divide up evenly between the mutated strains?
		tvec[mix][0] /= 2.0
		tvec[mix].insert (1, tvec[mix][0])

	assert abs(sum(tvec[mix])-1.0) < 1e6

	# Special case: weights flip in C strains? DNA
	if exp == 'C' and 'MIT3556' in inp:
		exp = 'C2'

	title,fields  = convert_name(inp)
	return truth[exp], tvec[mix], title, fields



#from ml_common import *
#yearmap = {}



# PARAMETERS

krange = tuple(range(1,6+1))
gamma = 0.01 # Std.dev of noise
s = StrainRecon()
num_snps = 24

num_cores = multiprocessing.cpu_count()

memory = Memory('pycache',verbose=0)



def read_field_data_year(fn, year=None):
	delim = ','
	with open(fn, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delim)
		next(csvreader, None) # Skip header
		for row in csvreader:
			# "Sample_107_ZHO5553A19" 0 1 0.26739214924213 1 0.868887749871861 0.866425276523982 0.739234808702176 0 0.186329460013671 1 1 0.166319396212437 0.364131531168045 0.882575076425103 0.0304738050511375 0.209611913357401 0.747326732673267 0.753568030447193 0.830925785133634 0.351467611336032 0.664400194269063 0.141422456371419 0 1
			sample = row[0].replace('"','')
			#yr = yearmap[sample]
			#yr = re.search (r"Sample_([0-9]*)_", sample)
			yr = re.search (r"_([0-9]*)", sample[-5:])

			if year is not None:
				if  yr is not None:
					yr = yr.group(1)
					if yr not in year:
						#print ("Filtering out %s != %s" % (yr,year[0]))
						continue # Filter out other years
					#print ("Found year {} in {}".format (year, sample))
				else:
					#print ("Uh oh")
					continue
				
			x = [ float(z) for z in row[1:] if z != "NA" ]

			yield x,yr,sample

def read_pilot_data(fn):
	delim = ','
	with open(fn, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delim)
		next(csvreader, None) # Skip header
		for row in csvreader:
			# "73","Sample_A1_4NOPA_MIT3556A38",0.0358675659104844,0,0,0.0218435998252512,1,0.0230057293962098,0.973744583227122,1,1,0.962487914921044,0.96451192668793,0.980318257956449,1,NA,0,1,0.0351089588377724,NA,0.0216093037875216,1,1,0,NA,1

			tr,tv,sample,_ = infertruth (row[0].replace('"',''))

			x = [ float(z) for z in row[1:] if z != "NA" ]

			yield x,tr,sample

def process(tup,gamma):
	x,yr,sample = tup
	# Run the misfits calculation
	print ("Misfitting for {}".format(sample))
	mf = s.misfits(x, krange, gamma=gamma)
	return x,mf,yr,sample  


cached_process = memory.cache(process)


#### MAIN #####
if __name__ == "__main__":
	log.basicConfig(stream=sys.stderr, level=log.DEBUG)

	field_fn = "data/FrequenciesField26June2019.csv"
	pilot_fn = "data/FrequenciesLab26June2019.csv"

	# Field data
	years = [ "1996", "2001", "2007", "2012" ]
	fd = list(read_field_data_year(field_fn,years))
	pl = list(read_pilot_data(pilot_fn))

	resolution = 1e-5
	#arg = float(sys.argv[1])


	for fudge in np.logspace(np.log10(1e-8),np.log10(1.0),300):
		shouldbeone    = 0
		areone         = 0
		shouldnotbeone = 0
		arenotone      = 0
		crazy_pilot    = []
		pilot          = []

		for x,mf,yr,sample in Parallel(n_jobs=num_cores)(delayed(cached_process)(tup,gamma) for tup in fd + pl):
			mf = np.array(mf)

			diverse = len([xx for xx in x if xx > resolution and xx < 1.0-resolution])
			# Find first drop below threshold
			thres = fudge
			best = sum(mf >= thres)

			if best+1 <= 1:
				areone += 1	
				if diverse > 1:
					shouldnotbeone += 1
			else:
				arenotone += 1
				if diverse <= 2:  #maybe account for original length?
					shouldbeone += 1

			# Assess pilot data
			if yr not in years:
				# tr = infertruth(sample)
				tr = yr #Truth stored here
				#if best+1 != len(tr): # strains should be normally 3
				if abs(best+1 - len(tr)) > 1: # strains should be normally 3
					crazy_pilot += [ best+1 ]
					#tqdm.write ("Name:\t{}x:\t{}\nMisfit:\t{}\nLength:\t{}\tDiverse:\t{}\tEstimated k:\t{}\tTrue k:\t{}\tMF for k=1:\t{}\n".format (sample, x, mf, len(x), diverse, best+1, len(tr), mf[0]))
					
				pilot += [ best+1 ]

		print ("Threshold\t{}\tFP\t{}\tFN\t{}\tNon-3 pilot\t{}\t{} +-{}".format (fudge, 100.0*shouldnotbeone/(areone+resolution), 100.0*shouldbeone/(arenotone+resolution),len(crazy_pilot),np.mean(pilot),np.std(pilot)))



