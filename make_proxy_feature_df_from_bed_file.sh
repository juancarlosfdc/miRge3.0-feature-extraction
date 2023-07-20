#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate mirge-dev

bn=$(basename $1 .bed)

bedtools getfasta -fi ~/gtex-smallrna-jcfc/refs/Homo_sapiens_assembly38_noALT_noHLA_noDecoy.fasta -bed $1 -s > pileups.fasta

samtools faidx pileups.fasta

python make_proxy_feature_df_from_fasta.py pileups.fasta.fai pileups.fasta $1 example_feature_df.tsv ${bn}_proxy_df.tsv 
