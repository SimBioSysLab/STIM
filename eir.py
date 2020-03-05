import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import matplotlib.ticker as mtick
import csv
import itertools

style.use('bmh')
plt.rc('svg', fonttype='none')
print (style.available)


year,eir,prev,prevb,preva,stim = np.loadtxt ('data/EIR-Kenya.csv', delimiter="\t", unpack=True)

plt.rc('ytick', labelsize=9)

fig, axs =  plt.subplots(2, 2, sharex='col', gridspec_kw={'hspace': 0.1}, figsize=(10,6)) 


for i, label in enumerate(('A', 'C', 'B', 'D')):
    ax = axs.flat[i]
    ax.text(-0.15 if i % 2 == 0 else 1.1, 0.95, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')



plt.xlim(1995, 2013)

mask = np.isfinite(eir.astype(np.double))
ax = axs[0][0]
ax.set(xlim=(1995, 2013))
ax.set(xticks=[1996,2001,2007,2012])
ax.plot(year[mask], eir[mask], 'g--', zorder=-1,lw=1)
ax.plot(year, eir, 'go-',zorder=0)
#plt.ylabel ('Entomological Inoculation Rate (EIR)')
ax.set(ylim=(0,80))
ax.set(ylabel='EIR')
ax.label_outer()

ax = axs[1][0]
mask = np.isfinite(prev.astype(np.double))
ax.set (xlim=(1995, 2013))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.errorbar(year,prev*100, yerr=100.0*(preva-prev),zorder=-1) #-eir)
ax.plot(year[mask], prev[mask] * 100.0, 'r--', zorder=-2, lw=1)
ax.plot(year, prev * 100.0, 'ro-', zorder=0)
ax.set(ylim=(0,100))
ax.set(ylabel='Prevalence')
ax.set(xlabel='Survey year')
ax.label_outer()




mask = np.isfinite(stim.astype(np.double))
ax = axs[0][1]
ax.set(ylim=(0,5))
ax.set(ylabel='MOI (average)')

years = [ "1996", "2001", "2007", "2012" ]
bins = np.arange (len(years)+1) - 0.5

ax.set_xticks (range(4))
ax.set (xlim=[-0.5,3.5])
ax.set (ylim = [0,5])
x = stim[mask] #list(stim[mask])

weights = [100.0*xx for xx in x]

print (stim[mask])

patches = ax.bar(years, x, width=0.5, color=["#a0b0ff"]) #, label=["%s%d strain%s" % ("=" if v != krange[-1] else "$\geq$", v, "s" if v != krange[0] else "") for v in krange]) #, hatch=hatch)

for i,p in enumerate(patches):
	#l = np.sum(np.array(byyear[i]) == len(patches)-j-1)
	h1 = p.get_height()
	print ("{} {}".format(p,h1))
	z = x[i] #100.0 #* l / float(sum(lk[years[i]].values()))
	ax.text (p.get_x() + p.get_width() / 2., p.get_y() + h1 / 2., "%2.2f" % z, ha="center", va="center", color="black", fontsize=8, fontweight="normal")


#Year 1996	1=2	2=2	3=6	4=19	5=36	6=0	Average (including 5+):	4.3077	Average (excluding 5):	3.4483	Median:	5.0000
#Year 2001	1=3	2=4	3=10	4=27	5=27	6=0	Average (including 5+):	4.0000	Average (excluding 5):	3.3864	Median:	4.0000
#Year 2007	1=6	2=2	3=11	4=18	5=11	6=0	Average (including 5+):	3.5417	Average (excluding 5):	3.1081	Median:	4.0000
#Year 2012	1=13	2=6	3=14	4=32	5=12	6=0	Average (including 5+):	3.3117	Average (excluding 5):	3.0000	Median:	4.0000

def barch(plt):
	years = [ "1996", "2001", "2007", "2012" ]
	krange = tuple(range(1,6+1))
	def ind(yr):
		return years.index (yr)

	hm = {}
	hm[1] = [0] * 2 + [1] * 3 + [2] * 6 + [3] * 13
	hm[2] = [0] * 2 + [1] * 4 + [2] * 2 + [3] * 6
	hm[3] = [0] * 6 + [1] * 10+ [2] * 11+ [3] * 14
	hm[4] = [0] * 19+ [1] * 27+ [2] * 18+ [3] * 32
	hm[5] = [0] * 36+ [1] * 27+ [2] * 11+ [3] * 12
	hm[6] = [0] * 0 + [1] * 0 + [2] * 0 + [3] * 0
	m = [hm[v] for v in krange]
	total = { 0 : 2+2+6+19+36+0, 1 : 3+4+10+27+27+0, 2 : 6+2+11+18+11+0, 3 : 13+6+14+32+12+0 }

	colors = ["#88ccee", "#44aa99", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499" ]
	#colors = ["#222222", "#666666", "#aaaaaa", "#eeeeee", "#dddddd", "#882255", "#AA4499" ]
	colors = colors[0:len(krange)]


	plt.set(xlabel='Survey year')
	plt.set(ylabel ='MOI (across samples)')
	plt.set_xticks(range(4))
	plt.set_xticklabels (years)
	plt.set (xlim=[-0.5,3.5])
	plt.yaxis.set_major_formatter(mtick.PercentFormatter())
	plt.set (ylim = [0,100])

	weights = np.array ([[100.0 / float(total[int(y)]) for y in hm[v]] for v in krange])
	bins = np.arange (len(years)+1) - 0.5
	hatch='/'
	_, _, patches = plt.hist(m, bins=bins, histtype='bar', stacked=True, weights=weights, rwidth=0.5, color=colors, label=["%d%s strain%s" % (v, "" if v != krange[-1] else "+", "s" if v != krange[0] else "") for v in krange]) #, hatch=hatch)
	plt.legend (bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, prop={'size':10}, )

	mm = np.array(m)
	lk = { year : { v : len(list (filter (lambda y:y==ind(year), hm[v]))) for v in krange } for year in years }
	for j,bc in enumerate(patches):
		for i,p in enumerate(bc):
			l = lk[years[i]][krange[j]]
			if l == 0:
				continue
			h1 = p.get_height()
			print ("{} {}".format(p,l))
			z = 100.0 * l / float(sum(lk[years[i]].values()))
			plt.text (p.get_x() + p.get_width() / 2., p.get_y() + h1 / 2., "%d%%" % int(z), ha="center", va="center", color="black", fontsize=8, fontweight="normal")


ax = axs[1][1]
barch (ax)


for i,j in itertools.product (range(2), range(2)):
	axs[i][j].spines['top'].set_visible(False)
	axs[i][j].spines['right'].set_visible(False)

plt.savefig('figures/eir.pdf', bbox_inches='tight')

plt.savefig('figures/eir.svg', bbox_inches='tight')

plt.show()


