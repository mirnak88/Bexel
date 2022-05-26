import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib
from pathlib import PosixPath
import data

## Ucitavanje podataka iz fajla
def data_import():
    data_dir= pathlib.Path.cwd().parent
    data_file = str(data_dir)+'\data'+ "\Model01_v2.csv"
    data = pd.read_csv(str(data_file), header=[3])
    return data



def data_labeling(data):
    ## Definisanje Labela
    labels = ['Bounding Box Height', 'Bounding Box Length', 'Bounding Box Width', 'Calculated Gross Area',
              'Calculated Surface Area', 'Calculated Volume', 'Max Z', 'Min Z', 'Triangle Count', 'H/L', 'H/W', 'L/W',
              'Target Category']
    labels_X = ['Bounding Box Height', 'Bounding Box Length', 'Bounding Box Width', 'Calculated Gross Area',
                'Calculated Surface Area', 'Calculated Volume', 'Max Z', 'Min Z', 'Triangle Count', 'H/L', 'H/W', 'L/W']
    labels_Y = ['Target Category']

    df_data = data[data.columns.intersection(labels)]
    #Dodavanje novih velicina, odnos visina-duzina, visina-sirina, duzina -sirina
    df_data['H/L'] = df_data['Bounding Box Height'].div(df_data['Bounding Box Length'].values)
    df_data['H/W'] = df_data['Bounding Box Height'].div(df_data['Bounding Box Width'].values)
    df_data['L/W'] = df_data['Bounding Box Length'].div(df_data['Bounding Box Width'].values)

    X_data = data[data.columns.intersection(labels_X)]
    Y_data = data[data.columns.intersection(labels_Y)]
    return  X_data,Y_data,df_data

def correlation_matrix(df_data):
    mapping = {'CONDUIT': 1, 'COVERING': 2, 'HARDSCAPE': 3, 'MASS (CUT)': 4, 'MASS (FILL)': 5, 'PLANTING': 6, 'ROAD': 7,
               'SLAB': 8, 'SITE': 9}
    df_data = df_data.applymap(lambda s: mapping.get(s) if s in mapping else s)
    corr = df_data.corr()
    f = plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, cmap="coolwarm");
    plt.show()
    return df_data

def plot_box(labels_X, labels_Y, df_data):
    fig, ax = plt.subplots(round(len(labels_X)/2))
    fig2, ax2 = plt.subplots(len(labels_X)-round(len(labels_X)/2))
    for it,label in enumerate(labels_X):
        if it<round(len(labels_X)/2):
            sns.boxplot(x=labels_Y[0], y=label, notch=True, ax=ax[it], showfliers=False, data=df_data);
        else:
            sns.boxplot(x=labels_Y[0], y=label, notch=True, ax=ax2[it-round(len(labels_X)/2)], showfliers=False, data=df_data);

    plt.show()

data=data_import()
X_data,Y_data,df_data=data_labeling(data)

## Definisanje Labela
labels = ['Bounding Box Height', 'Bounding Box Length', 'Bounding Box Width', 'Calculated Gross Area',
          'Calculated Surface Area', 'Calculated Volume', 'Max Z', 'Min Z', 'Triangle Count', 'H/L', 'H/W', 'L/W',
          'Target Category']
labels_X = ['Bounding Box Height', 'Bounding Box Length', 'Bounding Box Width', 'Calculated Gross Area',
            'Calculated Surface Area', 'Calculated Volume', 'Max Z', 'Min Z', 'Triangle Count', 'H/L', 'H/W', 'L/W']
labels_Y = ['Target Category']


# Crtanje matrice correlacija, kojoj je prethodilo mapiranje izlazne labele na brojene vrednosti
correlation_matrix(df_data)

# Broj elementa po kategorijama
print(df_data.groupby(["Target Category"]).size())

# Prosecna vrednost po kategorijama
print(df_data.groupby(["Target Category"]).mean())

# Izbacivanje SITE kao kategorije jer ima samo jedan uzorak
df_data = df_data[df_data["Target Category"] != "SITE"]
print(df_data.groupby(["Target Category"]).size())

plot_box(labels_X, labels_Y, df_data)



