#!/bin/bash
set -e
if [ -z "$JAVA_HOME" ]; then
    echo "ERROR!! Set environment variable JAVA_HOME to point to your java location (e.g., /usr/lib/[*your_java_folder*])"
    exit 1
fi  
mkdir -p ./berkeleyparser/
wget https://github.com/emcnany/berkeleyinterface/archive/master.zip -O ./install/berekeleyinterface.zip
wget https://github.com/astraw/svg_stack/archive/master.zip -O install/svg_stack.zip
wget "http://downloads.sourceforge.net/project/jpype/JPype/0.5.4/JPype-0.5.4.2.zip?r=&ts=1459330071&use_mirror=nchc" -O install/JPype-0.5.4.2.zip
wget "https://github.com/slavpetrov/berkeleyparser/raw/master/eng_sm6.gr" -O ./berkeleyparser/eng_sm6.gr
wget "https://github.com/slavpetrov/berkeleyparser/raw/master/BerkeleyParser-1.7.jar" -O ./berkeleyparser/BerkeleyParser-1.7.jar
wget https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.3.1/stanford-corenlp-3.3.1.jar -O ./dependency_tree/stanford-corenlp-3.3.1.jar
cd install/
unzip -o svg_stack.zip 
unzip -o berekeleyinterface.zip
unzip -o JPype-0.5.4.2.zip
cd berkeleyinterface-master/
# fix version compatibility issue between PropS, Berkeley Interface and setuptools
# permanent fix should happen in Berkeley Interface repository, temporary fix until then:
sed -i s/six==1.5.2/six==1.10.0/ requirements.txt
python setup.py install
cd ../svg_stack-master/
python setup.py install
cd ../JPype-0.5.4.2/
python setup.py install
cd ../
pip install -r requirements.txt
cd ../../
export PYTHONPATH=${PYTHONPATH}:`pwd`
