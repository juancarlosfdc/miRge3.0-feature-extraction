import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def parse_coordinate_string(row):
    row['chr'] = re.search('[^:]*', row[0]).group(0)
    row['startPos'] = int(re.search('(?<=:)[^-]*', row[0]).group(0))
    row['endPos'] = int(re.search('(?<=-)[^(]*', row[0]).group(0))
    return row

def add_sequence(row, fasta_string):
    row['sequence'] = fasta_string[row[2]:row[2]+row[1]]
    return row

def add_cluster_name(row, bed):
    formatted_coords = re.sub('(\(|\))', '', row['pileup_coords'])
    formatted_coords = re.sub('(?<=[0-9])-(?=[0-9])', '_', formatted_coords)
    row['clusterName'] = 'pileup_id:' + row['pileup_id'] + ':' + formatted_coords
    return row
    
def change_format_to_mirge(df, proxy_df):
    for col in proxy_df.columns:
        if 'Seq' in col:
            df[col] = df['sequence']
        elif col not in df.columns and len(set(proxy_df[col])) == 1:
            # this is a dummy column in the proxy feature df, so port over the dummy variables
            df[col] = proxy_df[col].values[0]
    return df[proxy_df.columns]

def faidx_to_feature_df(faidx, fasta, bed, proxy_df):
    proxy_df = pd.read_csv(proxy_df, sep='\t')
    faidx = pd.read_csv(faidx, sep='\t', header=None, names=['pileup_coords', 'length', 'offset', 'linebases', 'linewidth'])
    with open(fasta, 'r') as fasta:
        fasta_string = fasta.read()
    df = faidx.apply(add_sequence, fasta_string=fasta_string, axis=1)
    df = df.apply(parse_coordinate_string, axis=1)
    bed = pd.read_csv(bed, sep='\t', header=None, names=['chr', 'startPos', 'endPos', 'pileup_id', 'score', 'strand'])
    df = df.merge(bed.iloc[:, :4], on=['chr', 'startPos', 'endPos'])
    df = df.apply(add_cluster_name, bed=bed, axis=1)
    df = change_format_to_mirge(df, proxy_df)
    return df

df_ready_for_mirge = faidx_to_feature_df(faidx=sys.argv[1], fasta=sys.argv[2], bed=sys.argv[3], 
                                         proxy_df=sys.argv[4])

df_ready_for_mirge.to_csv(sys.argv[5], sep='\t', index=None)
