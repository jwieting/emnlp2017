#train paraphrastic sentence embedding models
if [ "$1" == "gran" ]; then
    sh train.sh train.py -model gran -gran_type 1 -min_value 0 -max_value 30 -data ../data/giga.fr.length
elif [ "$1" == "wordaverage" ]; then
    sh train.sh train.py -model wordaverage -margin 0.4 -min_value 0 -max_value 30 -data ../data/giga.fr.length

#train classification models
elif [ "$1" == "classification-wordaverage" ]; then
    sh train.sh train_translation_quality.py -model wordaverage -data ../data/giga.fr
elif [ "$1" == "classification-lstm" ]; then
    sh train.sh train_translation_quality.py -model lstm -data ../data/giga.fr
else
    echo "$1 not a valid option."
fi