#!/bin/bash
for filename in ../training_data/fastqs/*
do
    echo $filename
    /home/jfernand/anaconda3/envs/mirge-dev/bin/miRge3.0 -s $filename -lib ../miRge3_Lib -on human -db mirgenedb -o output_dir_may_3 -gff -nmir -trf -ai -cpu 16 -a illumina -pbwt /home/jfernand/anaconda3/envs/mirge-dev/bin
done
