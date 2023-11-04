infile=$1
outfile=$2
lang=$3

# BPE decoding (assuming you used the standard "@@" delimiter for BPE)
cat $infile | sed 's/@@ //g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
cat $infile | sed 's/▁/ /g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
