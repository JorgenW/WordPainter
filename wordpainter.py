'''
Created 2016 at NTNU.

Run this script to generate images based on the proceedure file.
'''


import runall
import os, datetime
import sys, glob

def wordpainter(full_run=False, net_params=["net_fn","param_fn","mean"], model_name="new"):
	if full_run:
		from standard_proceedure import *
		proceedures[test]["model"] = model_name
	else:
		from proceedure import *
	missing = {}
	for test in proceedures: #Check for missing images for generating
		if proceedures[test]["word"].lower() != "no guide":
			if len(glob.glob('words/'+proceedures[test]["word"]+'/*.jpg')) < proceedures[test]["numberOfImages"]:
				if proceedures[test]["word"] in missing: # if already found to be missing images
					if proceedures[test]["numberOfImages"] > missing[proceedures[test]["word"]]: # use largest number of missing images
						missing[proceedures[test]["word"]] = proceedures[test]["numberOfImages"]
				else:
					missing[proceedures[test]["word"]] = proceedures[test]["numberOfImages"]
	if len(missing) != 0:
		print("To complete the current proceedure schedule, guide images are needed. Please fetch the following words:")
		for missed in missing:
			print("The word " + missed + " needs a total of " + str(missing[missed]) + " images.")
		if "yes" == raw_input("Do you want to proceed and skip the missing words?(yes/*): ").lower():
			print("Proceeding with incomplete proceedure..")
		else:
			sys.exit()
	dir_session = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # folder name based on timestamp

	for test in proceedures:
		runall.RunAll(
			test,
			proceedures[test]["inputImage"],
			proceedures[test]["word"],
			proceedures[test]["numberOfImages"],
			proceedures[test]["database"],
			proceedures[test]["iterations"],
			proceedures[test]["octaves"],
			proceedures[test]["octaveScale"],
			proceedures[test]["model"],
			proceedures[test]["layer"],
			proceedures[test]["jitter"],
			proceedures[test]["stepSize"],
			proceedures[test]["mean"],
			proceedures[test]["focus"],
			dir_session, full_run=full_run, net_params=net_params)
