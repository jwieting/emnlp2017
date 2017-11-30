mkdir data
cd data

#download data, word embeddings, etc.
wget http://www.cs.cmu.edu/~jwieting/emnlp2017-demo.zip
unzip -j emnlp2017-demo.zip
rm emnlp2017-demo.zip

#filter giga.fr data by translation score, length, and 1-gram, 2-gram, and 3-gram overlap.
cd ../main
python filter_data.py -filtering_method trans -infile ../data/giga.fr -outfile ../data/giga.fr.trans
python filter_data.py -filtering_method length -infile ../data/giga.fr -outfile ../data/giga.fr.length
python filter_data.py -filtering_method ovl-1 -infile ../data/giga.fr -outfile ../data/giga.fr.ovl1
python filter_data.py -filtering_method ovl-2 -infile ../data/giga.fr -outfile ../data/giga.fr.ovl2
python filter_data.py -filtering_method ovl-3 -infile ../data/giga.fr -outfile ../data/giga.fr.ovl3