PYGMG -- HPGMG in Python
========================

This is a simple python reference implementation of the [HPGMG](https://hpgmg.org/) project 


[![Build Status](https://travis-ci.org/ucb-sejits/pygmg.svg?branch=master)](https://travis-ci.org/ucb-sejits/pygmg)
[![Coverage Status](https://coveralls.io/repos/ucb-sejits/pygmg/badge.svg?branch=master)](https://coveralls.io/r/ucb-sejits/pygmg?branch=master)

Quick Start
-----------
How to get things up and running from the command line.  This assumes you have
a working version python running and accessible from the shell and that you
have installed pip and it works too.
Given all that, from the shell:
```shell
git clone https://github.com/ucb-sejits/pygmg.git
cd pygmg
pip install -e .
```
You are now ready to run, first use the run script with --help to see the options.
And then do a simple run to see what happens.
```shell
./run_finite_volume --help
./run_finite_volume 4 -d 2 -nv 20
```
Using iPython
-------------
Once you have it running as above, a good way to visualize what is happening
is to us ipython.  From the command line.
```shell
ipython notebook
```
This creates a local web service running on localhost:8888  (on a laptop or desktop this
command will usually take you to the browser.  Select the notebooks folder and 
then try the SimpleSolver-HeatMap-2D notebook.
It is set up to show intermediate files to provide visual clues as to whether things
are working properly.
