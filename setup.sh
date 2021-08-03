#! /usr/bin/env bash

echo "Setting up paradigm-clusters"
echo

echo "#################################"
echo "creating conda python environment"

conda env create -n paradigm-clusters -f requirements.txt python=3.9
conda activate paradigm-clusters

echo "#################################"
echo "downloading Pitman-Yor Adaptor Grammar Sampler (PY-AGS)"

# in case of errors during make, check the README in py-cfg

wget http://web.science.mq.edu.au/~mjohnson/code/py-cfg-2013-09-23.tgz
tar -xvf py-cfg-2013-09-23.tgz
rm py-cfg-2013-09-23.tgz
cd py-cfg
make 
cd ..

echo "#################################"
echo "cloning MorphAGram repo"

git clone https://github.com/rnd2110/MorphAGram.git
cd MorphAGram
git checkout 0ba38b0fe0f2f966c490df9ed830ef8a976ce6f3 # set to working version at dev time
cd ..

echo "#################################"
echo "cloning SIGMORPHON Task 2 repo"
git clone https://github.com/sigmorphon/2021Task2.git
