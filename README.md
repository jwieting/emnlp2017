# emnlp2017

Code to train models from "Learning Paraphrastic Sentence Embeddings from Back-Translated Bitext".

The code is written in python and requires numpy, scipy, theano, and the lasagne libraries.

To get started, run setup.sh to download the required files such as training data and evaluation data. The script will also filter giga.fr (150k examples from English-French Gigaword) by translation score, length, and 1-gram, 2-gram, and 3-gram overlap. Back-translated data for the other corpora are available at http://www.cs.cmu.edu/~jwieting.

There is also a demo script that takes the model that you would like to train (either a paraphrastic sentence embedding model or a reference/translation classification model) as a command line argument (check the script to see available choices). Check main/train.py and main/train_translation_quality.py for command line options.

If you use our code or data for your work please cite:

@inproceedings{wieting-17-backtrans,
        author = {John Wieting, Jonathan Mallinson, and Kevin Gimpel},
        title = {Learning Paraphrastic Sentence Embeddings from Back-Translated Bitext},
        booktitle = {Proceedings of Empirical Methods in Natural Language Processing},
        year = {2017}
}
