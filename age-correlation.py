import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import re
import csv
import strainpycon

fn = "data/FrequenciesField9Aug2019.csv"
thres = 1.8e-7
gamma = 0.01
nrange = range(1,6+1)
sp = strainpycon.StrainRecon()


lookup = { "1996" : {}, "2007" : {}, "2001" : {}, "2012" : {}}

with open('data/ages-96-01-07.csv', 'r') as f:
	csvreader = csv.reader(f, delimiter=",")
	next(csvreader, None) #Header
	for row in csvreader:
		sample = row[0]
		age = row[4]
		year = row[1]
		#if year != "2007":
		#	continue

		if age == '' or age == '0':
			age = str(int(float(row[3]) * 12.0))
		lookup[year][sample] = age

		#print (sample, age, year, row)


		#samplecode,year,date,ageyrs,agemonths,matchid,INDIVIDID,LOCATIONID,HHID,Original_code,PMM3,Processed
		#22,1996,11/19/1996,,5.092402464,38,,,,17/087B,10426.66667,

		#sample = row[0].replace('"','')
		#yr = yearmap[sample]
		#yr = re.search (r"Sample_([0-9]*)_", sample)
		#yr = re.search (r"_([0-9]*)", sample[-5:])

with open('data/ages-12.csv', 'r') as f:
	csvreader = csv.reader(f, delimiter=",")
	next(csvreader, None) #Header
	for row in csvreader:
		sample = row[1]
		age = row[11]
		year = "2012"


		# Obs,SampleName,SampleCode,Smear,MaxAlleles,minalleles,Maxalleles2,Results,comp,DateTaken,Sex,AgeinMonth,AgeinYear,FeverLast2Weeks,Feverin24H,agecat3,HemocueReading,aspres1,aspeci1,asdens1,gmpres1,gmdens1,ITN,studyarea2,antimalarialtaken,latitude,longitude,altitude,NewSamplecode,Processed,year,samplecode1,Used,calcdensityifwbc,HIgh,randID,SELECTED,CalcDensityIfRBC
		#115,327,K5F7P,P,1,1,1,1,12-284,6/22/2012,Male,9,,2,,6mo- <5yo,8.6,Positive,PF,378,Negative,,No,Asembo,0,-0.190935,34.37265833,1145.2,,,,,,6048,,0.08479847,1,79380

		if age == '' or age == '0' or int(age) == 0:
			age = str(int(float(row[12]) * 12.0))

		#print (sample, age, year, row)

		lookup[year][sample] = age


### Process 

#print (lookup)

coll = { "all" : []}
#j = 0

with open(fn, 'r') as f:
	csvreader = csv.reader(f, delimiter=",")
	next(csvreader, None) #Header
	for row in csvreader:
		sample = row[0]
		#print (sample)

		unp = re.search (r"Sample_([0-9]*)_([^_]*)_([0-9]*)", sample)
		#print (sample, unp.groups())
		#sample_ID,_,yr = unp.groups()
		yr,sample_ID,_ = unp.groups()

		if not (yr in lookup and sample_ID in lookup[yr]):
			unp = re.search (r"Sample_([0-9]*)_(.*)_([0-9]*)$", sample)
			sample_ID,_,yr = unp.groups()
			
		if yr in lookup and sample_ID in lookup[yr]:
			bases = np.array ([ float(x) if x != "NA" else np.nan for x in row[1:]])	
			mf = sp.misfits (bases, nrange=nrange, gamma=gamma)
			moi = np.sum (mf > thres)+1
			#print ("FOUND\t{}\t{}\t{}\t{}".format(yr, sample_ID, lookup[yr][sample_ID], moi)) # row[1:]))
			if yr not in coll:
				coll[yr] = []
			coll[yr].append ( (float(lookup[yr][sample_ID]), float(moi)) )
			coll["all"].append ( (float(lookup[yr][sample_ID]), float(moi)) )

		else:
			#print ("Not found: {}".format(sample))
			pass


pval = []
for yr in sorted(coll.keys()) + ["all"]:
	a = np.array(coll[yr]).T
	#print (a)
	print ("{}:\tr={}\tp={}".format (yr, np.corrcoef (a[0], a[1])[0,1], pearsonr (a[0], a[1])[1]))

	#print ("R:\t{}".format (pearsonr (a[0], a[1])))
	pval.append (pearsonr (a[0], a[1])[1])


print ("FDR:\t{}".format (multipletests(pval, alpha=0.05, method='fdr_bh')))

