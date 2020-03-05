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
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp


memory = Memory('pycache-long',verbose=0)



# PARAMETERS

krange = tuple(range(1,6+1))
gamma = 0.01 # Std.dev of noise
s = StrainRecon()
num_snps = 24

num_cores = multiprocessing.cpu_count()




def read_field_data_year(fn, year=None):
	delim = ','

	taken = {} 
	with open(fn, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delim)
		next(csvreader, None) # Skip header
		for row in csvreader:
			sample = row[0].replace('"','')
			tk = re.search (r"Sample_([0-9]*)_([0-9]*)", sample)
			if not tk:
				continue

			# July 25, duplicates in plate 6678 to be filtered out
			if "6678" in sample:
				print (tk.group(0))
				taken[tk.group(0)] = 1
				continue


	with open(fn, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delim)
		next(csvreader, None) # Skip header
		for row in csvreader:
			sample = row[0].replace('"','')

			# July 25, duplicates in plate 6678 to be filtered out
			if False:
				# FILTER _OUT_ 6678
				if "6678" in sample:
					print ("Filtering out from plate 6678 ({})".format(sample))
					continue
			else:
				# FILTER _IN_ 6678
				tk = re.search (r"Sample_([0-9]*)_([0-9]*)", sample)
				if "6678" not in sample and tk and tk.group(0) in taken.keys():
					print ("Filtering away ({}) to include plate 6678".format(sample))
					continue

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
				
			x = [ float(z) if z != "NA" else np.nan for z in row[1:]  ]

			yield x,yr,sample


def field_longitudinal(fn, thres):

	pp = PdfPages('figures/field_longitudinal_gamma={}_thres={}.pdf'.format(gamma,thres))

	years = [ "1996", "2001", "2007", "2012" ]

	def ind(yr):
		return years.index (yr)

	fig, ax = plt.subplots(1,1)
	ax.grid(False)

	# color-blind spectrum: http://personal.sron.nl/~pault/colourschemes.pdf
	colors = ["#88ccee", "#44aa99", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499" ]
	colors = colors[0:len(krange)]

	bars = []
	rows = []

	# Crude

	hm = { v: [] for v in krange}

	sm = 0
	total  = { ind(y) :  0 for y in years }
	byyear = { ind(y) : [] for y in years }

	def process_vc(tup):
		x,yr,sample = tup
		# Run the misfits calculation
		mf = s.misfits(x, krange, gamma=gamma)
		sr = { k : s.compute(x, k, gamma=gamma) for k in krange }
		return x,mf,yr,sample,sr 

	cached_process = memory.cache(process_vc)

	stf = open ('figures/strains.out','w')

	for year in tqdm(years):
		fdata = Parallel(n_jobs=num_cores)(delayed(cached_process)(tup) for tup in read_field_data_year(fn,[year]))

		for x,mf,yr,sample,sr in fdata:
			mf = np.array(mf)
			# Find first drop below threshold
			tqdm.write ("Length:{}. Interesting bases:{}. Misfits: {}".format(len(x),len([xx for xx in x if 1.0-1e-6 > xx > 1e-6]), mf))
			best = sum(mf > thres)
			if best >= len(mf):
				continue
			best += 1	
			if best not in hm:
				hm[best] = []
			byyear[ind(year)] += [ best ]
			hm[best] += [ ind(year) ]
			total[ind(year)] += 1

			# Print out the strain sequences of at least 5% proportion in a sample
			assert best in sr
			print (sr[best][1], max(sr[best][1]))
			stf.write ("DOMSTR\t{}\t{:.2f}%\n".format (yr, max (sr[best][1])))
			for i,f in enumerate(sr[best][1]):
				if f >= 0.05:
					stf.write ("STRAIN\t{}\t{}\t{}\t{:.2f}%\t{}\n".format (sample, yr, i, 100.0*sr[best][1][i], "".join(["{}".format(int(z)) if not np.isnan(z) else "N" for z in sr[best][0][i]])))

	stf.close()


	of = open('figures/field_longitudinal_gamma={}_thres={}.txt'.format(gamma,thres), 'w')
	for year in years:
		of.write("Year %s\t" % year + "\t".join(["%d=%d" % (v, len(list (filter (lambda y:y==ind(year), hm[v])))) for v in krange]))
		of.write ("\tAverage (including 5+):\t" + "%2.4f" % (np.mean(byyear[ind(year)])) +"\tAverage (excluding 5):\t" + "%2.4f" % (np.mean([sc for sc in byyear[ind(year)] if sc < 5])))
		of.write ("\tMedian:\t" + "%2.4f" % (np.median(byyear[ind(year)])) + "\n")
	m = [hm[v] for v in krange]

	plt.ylabel ('% samples')
	plt.xlabel ('Survey year')
	plt.xticks (np.arange(len(years)), years)
	plt.ylim([0,100])

	weights = np.array ([[100.0 / float(total[int(y)]) for y in hm[v]] for v in krange])
	bins = np.arange (len(years)+1) - 0.5
	hatch='/'
	_, _, patches = plt.hist(m, bins=bins, histtype='bar', stacked=True, weights=weights, rwidth=0.5, color=colors, label=["%s%d strain%s" % ("=" if v != krange[-1] else "$\geq$", v, "s" if v != krange[0] else "") for v in krange]) #, hatch=hatch)
	plt.legend (bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, prop={'size':10}, )

	mm = np.array(m)
	lk = { year : { v : len(list (filter (lambda y:y==ind(year), hm[v]))) for v in krange } for year in years }
	for j,bc in enumerate(patches):
		for i,p in enumerate(bc):
			#l = np.sum(np.array(byyear[i]) == len(patches)-j-1)
			l = lk[years[i]][krange[j]]
			if l == 0:
				continue
			h1 = p.get_height()
			print ("{} {}".format(p,l))
			z = 100.0 * l / float(sum(lk[years[i]].values()))
			ax.text (p.get_x() + p.get_width() / 2., p.get_y() + h1 / 2., "%d%%" % int(z), ha="center", va="center", color="black", fontsize=12, fontweight="bold")

	
	pp.savefig(bbox_inches="tight")
	pp.close()

	for y in years:
		of.write ("%s: length %d\n" % (y, len(byyear[ind(y)])))
	of.write ("{}\n".format (byyear[ind("1996")]))

	of.write("H1\t{}\t1996 vs 2001:\t{}\n".format(thres,sts.mannwhitneyu (byyear[ind("1996")], byyear[ind("2001")])))
	of.write("H2\t{}\t2007 vs 2012:\t{}\n".format(thres,sts.mannwhitneyu (byyear[ind("2007")], byyear[ind("2012")])))

	x = [ byyear[ind(y)] for y in years ]
	#pc = sp.posthoc_conover(x, val_col='values', group_col='groups', p_adjust='holm')

	kr = sts.kruskal (*x)
	of.write ("Kruskal-Willis:\n{}\n".format(kr))

	pc = sp.posthoc_conover(x, val_col='values', group_col='groups', p_adjust='fdr_tsbky')
	of.write ("Conover:\n{}\n".format(pc))
	# Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
	cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
	heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
	sp.sign_plot(pc, **heatmap_args)

	of.close()






def vary_thresholds(fn, thresholds, cthres):

	pp = PdfPages('figures/vary_thresholds_gamma={}.pdf'.format(gamma))

	years = [ "1996", "2001", "2007", "2012" ]

	def ind(yr):
		return years.index (yr)

	fig, ax = plt.subplots(1,1)
	ax.grid(False)

	# color-blind spectrum: http://personal.sron.nl/~pault/colourschemes.pdf
	colors = ["#88ccee", "#44aa99", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499" ]

	bars = []
	rows = []

	# Crude

	sm = 0
	hm     = { t : { v: [] for v in krange} for t in thresholds }
	total  = { t : { ind(y) :  0 for y in years } for t in thresholds }
	byyear = { t : { ind(y) : [] for y in years } for t in thresholds }
	compy = [ (years[k],years[k+1]) for k in range(len(years)-1) ]

	def process(tup):
		x,yr,sample = tup
		# Run the misfits calculation
		mf = s.misfits(x, krange, gamma=gamma)
		return x,mf,yr,sample  

	cached_process = memory.cache(process)

	for year in tqdm(years):
		fdata = Parallel(n_jobs=num_cores)(delayed(cached_process)(tup) for tup in read_field_data_year(fn,[year]))

		for x,mf,yr,sample in fdata:
			mf = np.array(mf)
			# Find first drop below threshold
			tqdm.write ("Length:{}. Interesting bases:{}. Misfits: {}".format(len(x),len([xx for xx in x if 1.0-1e-6 > xx > 1e-6]), mf))
			for thres in thresholds:
				best = sum(mf > thres)
				if best >= len(mf):
					continue
				best += 1	
				if best not in hm[thres]:
					hm[thres][best] = []
				byyear[thres][ind(year)] += [ best ]
				hm[thres][best] += [ ind(year) ]
				total[thres][ind(year)] += 1

	of = open('figures/vary_thresholds_gamma={}.txt'.format(gamma), 'w')
	x = {}
	bw = { }
	kruskals = {}
	conovers = { (a,b) : {} for (a,b) in compy }
	for thres in tqdm(thresholds):
		x[thres] = [ byyear[thres][ind(y)] for y in years ]

		try:
			kr = sts.kruskal (*x[thres])
			kruskals[thres] = kr[1] # p-value
			
			of.write ("\n{}\n{}\tKruskal-Willis:\n{}\n".format("*"*80,thres,kr))

			pc = sp.posthoc_conover(x[thres], val_col='values', group_col='groups', p_adjust='fdr_tsbky')
			for (a,b) in compy:
				#tqdm.write ("Conovers:\t{}\n".format( (a,b,ind(a),ind(b) )))
				#tqdm.write ("Conovers:\t{}\n".format(pc))
				#tqdm.write ("Conovers:\t{}\n".format(pc[ind(a)+1][ind(b)+1]))
				conovers[(a,b)][thres] = pc[ind(a)+1][ind(b)+1]
			
			of.write ("{}\tConover:\n{}\n".format(thres,pc))
		except:
			tqdm.write ('Exception with threshold {}, NaNing'.format(thres))

			of.write ("{}\tKruskal-Willis:\nNaN\n".format(thres))
			of.write ("{}\tConover:\nNaN\n".format(thres))
			kruskals[thres] = float('nan')
			for (a,b) in compy:
				conovers[(a,b)][thres] = float('nan')

	plt.ylabel ('$q$-value')
	plt.xlabel ('MOI Misfit Threshold ($T$)')
	plt.yscale('log')
	plt.xscale('log')

	plt.plot(thresholds, [kruskals[t] for t in thresholds], color='orange', linewidth=1.3, alpha=0.7, label='Kruskal-Willis')
	for i,(a,b) in enumerate(compy):
		plt.plot(thresholds, [conovers[a,b][t] for t in thresholds], color=colors[i], alpha=0.7, label='Conover-Imam %s vs. %s' % (a,b))
	plt.legend (prop={'size':10},loc='lower right')


	plt.axvline(x=cthres, color='k', linestyle='--', linewidth=0.5, label='', alpha=0.8)
	plt.text(cthres, 1e-5, 'Threshold used', horizontalalignment='left', size='small', color='k', alpha=0.8)

	plt.axhline(y=0.05, color='b', linestyle='--', linewidth=0.5, label='0.05', alpha=0.7)
	plt.text(cthres, 0.05, '$q$=0.05', horizontalalignment='right', size='small', color='b', alpha=0.7)


	pp.savefig()
	pp.close()

	of.close()






#### MAIN #####
if __name__ == "__main__":
	log.basicConfig(stream=sys.stderr, level=log.DEBUG)

	# Usage: field_longitudinal.py  [threshold]

	thres = 1.8e-7
	should_vary = False
	fn = "data/FrequenciesField9Aug2019.csv"

	if len(sys.argv) > 1:
		thres = float(sys.argv[1])

	field_longitudinal (fn, thres)
	if should_vary:
		vary_thresholds (fn, np.logspace(np.log10(thres)-1,np.log10(thres)+1,400), thres)




