# -*- coding: utf-8 -*-
#
# This file is part of my personal analysis kit.
#
# This kit is open-source software for analysing all sorts of data, mostly
# collected in the field of Experimental Psychology.
#
# Copyright (C) 2014, Edwin S. Dalmaijer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

__author__ = u"Edwin Dalmaijer"


# native
import math
import multiprocessing
import time

# external
import numpy

# custom
import helpers


def permutation_test(x, y, twotailed=True, maxperm=None, maxcpu=None):
	
	"""
	desc:
		Performs a permutation test of datasets x and y, calculating the T
		statistic (difference between means), the p statistic (proportion of
		all permutations where the absolute T was greater than or equal to
		the absolute T of the observed values, in a two-tailed test), as well
		as the amount of permutations and the maximal T value for a
		permutation (this can be used for corrections of multiple
		comparisons).
	
	arguments:
		x:
			desc:Dataset 1.
			type:NumPy Array
		y:
			desc:Dataset 2.
			type:NumPy Array
	
	keywords:
		twotailed:
			desc:Boolean indicating whether the test should be one or two
				sided. In a one-tailed test, the p-value is defined as the
				proportion of sampled permutations where T (difference in
				means) is greater than or equal than T for x and y. In a
				two-tailed test, p is the proportion of all permutations
				where the absolute T was greater than or equal to the
				absolute T of the observed values. (default = True)
			type:bool
		maxperm:
			desc:The maximal amount of permutations, or None for no cap.
				(default = None)
			type:int
		maxcpu:
			desc:The maximal amount of CPU cores that can be used in
				parallel to perform this calculation, or None to autodetect
				all CPUs. (default = None)
			type:int
	
	returns:
		A [T, p, Nperm, Tmax], where T is the difference in means of x and y,
		p is the proportion ofp is the proportion of all permutations where
		the absolute T was greater than or equal to the absolute T of the
		observed values (in a two-tailed test), Nperm is the amount of
		performed permutations, and Tmax is the maximal difference between
		the means in a permutation.
	"""
	
	# OBSERVED VALUES
	# combine the observed values
	obs = numpy.concatenate((x,y))
	# create a design matrix
	dm = numpy.concatenate((numpy.zeros(len(x)),numpy.ones(len(y))))
	# calculate T
	obsT = numpy.mean(obs[dm==0]) - numpy.mean(obs[dm==1])
	
	# MULTIPROCESSING
	# only do this if we're allowed more than one core
	if maxcpu == 1:
		cpus = 1
	else:
		# get the available CPUs
		cpus = multiprocessing.cpu_count()
		# correct if it's more than allowed
		if maxcpu < cpus and maxcpu != None:
			cpus = maxcpu
	
	# PERMUTATIONS
	# calculate the amount of permutations
	Nallperms = math.factorial(len(dm))
	if maxperm != None and Nallperms > maxperm:
		Nperms = maxperm
	else:
		Nperms = Nallperms
	
	# SLICES
	# calculate slice size
	if cpus > 1:
		slicesize = int(numpy.floor(Nperms / float(cpus)))
		maxslicesize = int(numpy.floor(Nallperms / float(cpus)))
	else:
		slicesize = Nperms
		maxslicesize = slicesize
	# create slices for all but the last CPU
	slices = []
	for i in range(cpus-1):
		# starting index number
		# ( "+ i * (maxslicesize-slicesize)" is to spread out the sampling
		# over all of the possible permutations)
		si = i * slicesize + i * (maxslicesize-slicesize)
		# add slice and completely separate permutation generator
		# (using only one leads to the separate slices iterating over that
		# single one, therefore shortening it, therefore rendering the slices
		# invalid!)
		slices.append([dm, si, si+slicesize])
	# add slice for final CPU, who will clean up the rest
	if len(slices) == 0:
		si = 0
	else:
		si = (len(slices)) * slicesize
	slices.append([dm, si, Nperms])
	
	# RUN PERMUTATION SLICES
	# create a Queue to store values
	queue = multiprocessing.Queue()
	# starting time
	t0 = time.time()
	# run all subprocesses
	processes = []
	for s in range(len(slices)-1):
		processes.append(multiprocessing.Process(target=helpers._run_permutations, args=[queue, slices[s], obs, obsT, twotailed]))
		processes[s].start()
	# run last slice in main process
	helpers._run_permutations(queue, slices[-1], obs, obsT, twotailed)
	
	# WAIT FOR ENDING
	for s in range(len(slices)-1):
		processes[s].join()
	# ending time
	t1 = time.time()
	print("performed %d permutations in %.3f seconds" % (Nperms, t1-t0))
	
	# GET DATA
	# empty the Queue
	maxT = 0
	overT = 0
	while not queue.empty():
		# get data
		mt, ot = queue.get()
		# update values
		overT += ot
		if maxT < mt:
			maxT = 0+mt
		# sleep for a bit, to prevent calling queue.empty too soon
		time.sleep(0.01)
	# calculate p
	p = overT / float(Nperms)
	
	return obsT, p, Nperms, maxT
	
	