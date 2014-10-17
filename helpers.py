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


def _run_permutations(outqueue, permslice, obs, obsT, twotailed):
		
		"""
		desc:
			Runs through a slice of permutations, and returns
		"""
		
		import itertools
		import numpy
		
		# get permutations
		# (dm = design matrix, si = starting index, ei = ending index)
		dm, si, ei = permslice
		permutations = itertools.islice(itertools.permutations(dm), si, ei)
		
		# starting values
		maxT = 0
		overT = 0
		empty = False
		# run until iteration stops
		while not empty:
			try:
				# get the next permutation
				permdm = numpy.array(permutations.next())
				# calculate T
				permT = numpy.mean(obs[permdm==0]) - numpy.mean(obs[permdm==1])
				# update values
				if twotailed:
					if numpy.abs(permT) >= numpy.abs(obsT):
						overT += 1
				else:
					if permT >= obsT:
						overT += 1
				if permT > maxT:
					maxT = 0 + permT
			# if the iteration stopped, the slice of permutations is empty
			except StopIteration:
				empty = True
		
		# put values in the queue
		outqueue.put([maxT,overT])
		
		# return successfully
		return True

