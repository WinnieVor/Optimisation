#!/usr/bin/env python
# coding: utf-8

# INSA ROUEN <br>
# MS ESD 2019-2020 <br>
# Winnie VORIHILALA <br>

# # <center> TP OPTIMISATION - Mix électrique renouvelable en 2050 </center>

# <strong> Pré-requis </strong> <br>
# Données à télécharger sur Open Data Réseaux Energies https://opendata.reseaux-energies.fr/ : 
# - Données éCO2mix nationales consolidées et définitives (janvier 2012 à février 2020) <a href="https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-national-cons-def/information/?disjunctive.nature">(lien)</a><br>
# - Parc national annuel de production par filière (2008 à 2019) <a href="https://opendata.reseaux-energies.fr/explore/dataset/parc-prod-par-filiere/information/?sort=annee">(lien)</a><br>
#  

# <strong> Mix électrique renouvelable en 2050 </strong> <br>
# Fin 2018, la France a publié sa Stratégie Nationale Bas Carbone (SNBC), qui définit la manière
# d’atteindre les objectifs de neutralité carbone en France en 2050. En particulier, la SNBC décrit
# comment les usages électriques pourraient se développer pour atteindre la neutralité carbone.
# L’objectif est d’analyser comment répondre aux projections de demande de la SNBC en déterminant
# un mix électrique neutre en carbone, à l’horizon 2050. Ce système électrique sera composé de moyens
# de production renouvelables : fermes éoliennes, fermes solaires, centrales à cycle combiné au biogaz.

# # 1 - Chargement des librairies 

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly as py
py.offline.init_notebook_mode()

import pulp


# # 2- Chargement des données 

# In[108]:


data_dir ="/Users/winnievorihilala/Documents/INSA/OPTI/data/"
mix = pd.read_csv(data_dir + "eco2mix-national-cons-def.csv", sep=";", index_col=False, low_memory=False)
parc = pd.read_csv(data_dir + "parc-prod-par-filiere.csv", sep=";", index_col=False, low_memory=False)
print(mix.shape)
print(parc.shape)


# In[109]:


parc.head()


# In[110]:


mix.head()


# In[111]:


mix.columns


# In[112]:


parc.columns


# # Question 1 : création d'un profil de demande

# Afin que le système soit adapté aux variations de demande, il est nécessaire de déterminer un profil de demande, c’est-à-dire la manière dont la demande fluctue au cours d’une année.
# Par ailleurs, afin que le système soit dimensionné correctement, il est important qu’il soit adapté à une année comprenant des journées et des heures de forte consommation.

# # Question 1.1 - Récupération des données de consommation en 2012 et calcul de proportion

# Dans la base Eco2Mix, récupérer les données de consommation d’électricité de l’année 2012. Pour chaque heure de cette année, calculer la proportion de la consommation par rapport à la consommation annuelle, de manière à obtenir un profil (en %). <br>

# # a) Récupération des données de consommation d’électricité de l’année 2012

# In[31]:


date = mix['Date']
heure = mix['Heure']
date_et_heure = mix['Date et Heure']
consommation = mix['Consommation (MW)']

#print(mix.shape[0])

Date = []
Heure = []
Date_et_heure = []
Consommation = []

for i in range(0,mix.shape[0]) :
    if(('2012' in date[i])==True):
        #print(i)
        Date.append(date[i])
        Heure.append(heure[i])
        Date_et_heure.append(date_et_heure[i])
        Consommation.append(consommation[i])    

dfDate = pd.DataFrame(Date)
dfHeure = pd.DataFrame(Heure)
dfDate_et_heure = pd.DataFrame(Date_et_heure)
dfConsommation = pd.DataFrame(Consommation)

frames = [dfDate, dfHeure, dfDate_et_heure, dfConsommation]
df = pd.concat(frames, axis=1)
df.columns = ['Date', 'Heure', 'Date_et_heure','Consommation']
df = df.dropna()
df = df.sort_values(by=['Date_et_heure'])
print("Le dataframe df est de dimension", df.shape)
df.head()
#df.sample(10)


# On constate que la consommation est affichée par demi-heure.

# # b) Calcul de proportion

# Pour chaque heure de l'année 2012, calculons la proportion de la consommation par rapport à la consommation annuelle, de manière à obtenir un profil (en %)

# Une proportion p s'exprime de la manière suivante : <br>
# $$p = n/N$$ 
# avec n : effectif et N effectif total

# Créons un nouveau dataframe avec les consommations de chaque demi heure sommées, de manière à avoir les consommations par heure pour l'année 2012.

# In[32]:


i = 0
k = 0

Date = []
Heure = []
Date_et_heure = []
Consommation = []

while i<df.shape[0]/2:
    #print(i)
    #print(k)
    date = df.iloc[k,0]
    heure = df.iloc[k,1]
    date_et_heure = df.iloc[k,2]
    consommation = df.iloc[k,3] + df.iloc[(k+1),3]

    Consommation.append(consommation)
    Heure.append(heure)
    Date.append(date)
    Date_et_heure.append(date_et_heure)
    k = k+2
    i = i+1

dfConsommation = pd.DataFrame(Consommation)
dfHeure = pd.DataFrame(Heure)
dfDate = pd.DataFrame(Date)
dfDate_et_heure = pd.DataFrame(Date_et_heure)

frames = [dfDate, dfHeure, dfDate_et_heure, dfConsommation]
df_conso = pd.concat(frames, axis = 1)
df_conso.columns = ['Date','Heure','Date_et_heure','Consommation']
print("Le dataframe df_conso est de taille ",df_conso.shape)
df_conso.head()


# Déterminons le nombre d'heures de notre dataframe

# In[10]:


(df_conso.Heure).shape


# Nous devrons par conséquent avoir 8784 proportions de consommation correspondant à chaque heure. Calculons maintenant ces proportions.

# In[33]:


N = np.sum(df_conso.Consommation) #effectif total
Proportion = []

for i in range(0,df_conso.shape[0]):
    proportion = df_conso.Consommation[i]/N*100
    #var_arr = round(var,4) #limiter à 4 chiffres après la virgule
    Proportion.append(proportion)
    
#len(Proportion)
dfProportion = pd.DataFrame(Proportion)
dfProportion.columns = ['Proportion']
frames = [df_conso,dfProportion]
df_proportion = pd.concat(frames,axis=1)
print("Le dataframe df_proportion est de taille ",df_proportion.shape)
df_proportion.head()


# Nous constatons que la proportion de consommation par heure oscille aux alentours de 0,01 %.

# # Visualisation

# Visualisons cette proportion de consommation par heure

# In[439]:


h = df_proportion.iloc[:,1] #correspond à la colonne Heure de df_proportion
c = df_proportion.iloc[:,4] #correspond à la colonne Consommation de df_proportion

dfh = pd.DataFrame(h)
dfc = pd.DataFrame(c)
frames = [h,c]
df_viz= pd.concat(frames,axis=1)
print("Le dataframe df_viz est de taille ",df_viz.shape)
df_viz.head()


# In[441]:


fig = plt.figure(figsize=(19, 9))
plt.plot(df_viz.Proportion, linewidth=1)
plt.title("Evolution de la proportion de consommation d'électricité par heure sur l'année 2012", fontsize=22, fontweight="bold")
plt.xlabel('Heure', fontsize=18)
plt.ylabel('Proportion de consommation (%)', fontsize=18)
#plt.savefig('./df_viz.Consommation.png')


# On constate un pic de la consommation en début et en fin d'année, ce qui correspond à la période hivernale. En effet la consommation d'électricité à cette période est plus élevée qu'en milieu d'année lors de la saison estivale. On observe une tendance baissière entre les 2 pics. Cette tendance baissière atteint son optimum en milieu d'année, à la période estivale. La baisse de la consommation d'électricité à cette période de l'année s'explique en partie par le fait qu'on n'utilise plus de chauffage. Par aileurs cette période coïncide également avec les départs en vacances, ce qui implique une baisse de la consommation d'électricité au sein des foyers.

# # Question 1.2 : Conversion en TWh

# Appliquons ce profil à une consommation d’électricité annuelle de 600 TWh, de manière à obtenir pour chaque heure de cette année une consommation (en TWh). Quelles variations peut-on observer à différentes granularités temporelles (journée, semaine, mois, année) ?

# Rappel : <br>
# - 1 MW = $10^6$W <br>
# - 1TW = $10^{12}$W <br>
# - kWh : Le kilowatt-heure ou kilowattheure (symbole kW h, kW⋅h ou, selon l'usage, kWh) est une unité d'énergie. Un kilowatt-heure vaut 3,6 mégajoules. <br>
# Si de l'énergie est produite ou consommée à puissance constante sur une période donnée, l'énergie totale en kilowatts-heures est égale à la puissance en kilowatts multipliée par le temps en heures. Le kilowatt-heure est surtout utilisé pour l'énergie électrique, mais il l'est aussi pour facturer le gaz combustible et faire des bilans énergétiques. <br>
# <center><strong>Energie (en kWh) = Puissance (en kW) * temps (en heure)</strong></center>
# 

# # a) Conversion

# Pour effectuer cette conversion, nous allons multiplier chaque valeur correspondant à notre colonne proportion par 600TWh. Nous mettrons ces nouvelles valeurs dans une nouvelle colonne qui sera appelée Profil et que nous rajouterons à notre dataframe.

# In[153]:


Profil_TWh = []

for i in range(0,df_proportion.shape[0]):
    p = df_proportion.Proportion[i]*6000000
    Profil_TWh.append(p) 

Profil_TWh = pd.DataFrame(Profil_TWh)
Profil_TWh.columns = ['Profil_TWh']
frames = [df_proportion,Profil_TWh]
df_profil = pd.concat(frames,axis=1)
print("Le dataframe df_profil est de taille ",df_profil.shape)
df_profil.head()


# # b) Visualisation des variations par granularité temporelle

# ## b1 - Visualisation par jour et par mois sur l'année 2012

# In[442]:


mois = []

for i in range(0,df_profil.shape[0]):
    m = (df_profil.iloc[i,2])[0:7]
    mois.append(m)

mois = pd.DataFrame(mois)

mois.columns = ['Mois']

#print(mois.shape)
frames = [df_profil,mois]
df_viz = pd.concat(frames, axis=1)

print("Le dataframe df_viz est de taille ",df_viz.shape)
print("\nLe nombre d'éléments uniques dans la colonne mois est ",(df_viz.Mois.unique()).shape)
print("Le nombre d'éléments uniques dans la colonne Date (jours) est ",(df_viz.Date.unique()).shape)
print("Le nombre d'éléments uniques dans la colonne Heure est ",(df_viz.Heure.unique()).shape)

jour_viz = df_viz.loc[:,['Date', 'Profil_TWh']].groupby('Date').sum()
mois_viz = df_viz.loc[:,['Mois', 'Profil_TWh']].groupby('Mois').sum()
heure_viz = df_viz.loc[:,['Heure', 'Profil_TWh']].groupby('Heure').sum()


print("\nLe dataframe jour_viz est de taille ",jour_viz.shape)
print("Le dataframe mois_viz est de taille ",mois_viz.shape)
print("Le dataframe heure_viz est de taille ",heure_viz.shape)

df_viz.head()


# In[446]:


h1 = heure_viz.iloc[0:4,:]
h2 = heure_viz.iloc[5:25,:]
#print(h1)
#print(h2)
frames = [h1,h2]
h = pd.concat(frames, axis=0)
#print(h)


# In[358]:


fig = plt.figure(figsize=(19, 9))
plt.plot(jour_viz, linewidth=1)
plt.title("Evolution du profil de demande par jour sur l'année 2012", fontsize=22, fontweight="bold")
plt.xlabel('Jour', fontsize=18)
plt.ylabel('Profil (TWh)', fontsize=18)
#plt.savefig('./annee_masse.png')


# On observe la même tendance que par rapport au graphique ci-dessus relatif à l'évolution de la consommation d'électricité par heure sur l'année 2012. On observe ainsi 2 pics en début et en fin d'année correspondant à la saison hivernale et une pente baissière entre ces 2 pics qui atteint son point le plus bas en milieu d'année, ce qui correspond à la période estivale de l'année.

# In[357]:


fig = plt.figure(figsize=(19, 9))
plt.plot(mois_viz, linewidth=1)
plt.title("Evolution du profil de demande par mois sur l'année 2012", fontsize=22, fontweight="bold")
plt.xlabel('Mois', fontsize=18)
plt.ylabel('Profil (TWh)', fontsize=18)
#plt.savefig('./annee_masse.png')


# In[447]:


fig = plt.figure(figsize=(19, 9))
plt.plot(h, linewidth=1)
plt.title("Evolution du profil de demande par heure sur l'année 2012", fontsize=22, fontweight="bold")
plt.xlabel('Heure', fontsize=18)
plt.ylabel('Profil (TWh)', fontsize=18)
#plt.savefig('./annee_masse.png')


# Ce graphique donne plus de détails sur l'évolution de la consommation par mois sur l'année 2012. On peut ainsi observer que :
# - la consommation est en forte hausse de septembre à février pour atteindre un pic en février (cette période correspond à la saison hivernale). Cette hausse s'explique par le fait que l'on consomme plus d'électricité durant l'hiver (chauffage, lumières allumées plus tôt dans la journée, ...)
# - la consommation est en forte baisse de février à août pour atteindre un premier pic en juin puis un deuxième pic (le plus bas) en août (cette période correspond à la saison estivale). Cette baisse s'explique par le fait que l'on consomme moins d'électricité durant l'été (pas de chauffage, beaucoup de départ en vacances à l'étranger..)

# ## SUITE VISUALISATION

# In[437]:


#Line Plot
trace1 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc eolien (MW)'],
                mode = "lines+markers",
                name = "Parc éolien (MW)",
                marker = dict(color = 'rgb(0, 255, 85)'),
                text = ['Année, Parc éolien (MW)'])
trace2 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc solaire (MW)'],
                mode = "lines+markers",
                name = "Parc solaire (MW)",
                marker = dict(color = 'rgb(0, 247, 255)'),
                text = ['Année, Parc solaire (MW)'])
trace3 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc gaz (MW)'],
                mode = "lines+markers",
                name = "Parc gaz (MW)",
                marker = dict(color = 'rgb(255, 0, 140)'),
                text = ['Année, Parc gaz (MW)'])
trace4 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc bioenergie (MW)'],
                mode = "lines+markers",
                name = "Parc bioénergie (MW)",
                marker = dict(color = 'rgb(230, 0, 255)'),
                text = ['Année, Parc bioénergie (MW)'])
trace5 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc nucleaire (MW)'],
                mode = "lines+markers",
                name = "Parc nucléaire (MW)",
                marker = dict(color = 'rgb(245, 176, 216)'),
                text = ['Année, Parc nucléaire (MW)'])
trace6 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc thermique fossile (MW)'],
                mode = "lines+markers",
                name = "Parc thermique fossile (MW)",
                marker = dict(color = 'rgb(255, 247, 0)'),
                text = ['Année, Parc thermique fossile (MW)'])
trace7 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc fioul (MW)'],
                mode = "lines+markers",
                name = "Parc fioul (MW)",
                marker = dict(color = 'rgb(0, 162, 255)'),
                text = ['Année, Parc fioul (MW)'])
trace8 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc charbon (MW)'],
                mode = "lines+markers",
                name = "Parc charbon (MW)",
                marker = dict(color = 'rgb(255, 170, 0)'),
                text = ['Année, Parc charbon (MW)'])
trace9 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc hydraulique (MW)'],
                mode = "lines+markers",
                name = "Parc hydraulique (MW)",
                marker = dict(color = 'rgb(79, 194, 102)'),
                text = ['Année, Parc hydraulique (MW)'])
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]
layout = go.Layout(title = "Evolution de la quantité d'électricité produite par parc en France (2008 à 2018)",
              xaxis= dict(title= 'Année',ticklen= 5,zeroline= False),
              yaxis= dict(title= "Quantité d'électricité produite par parc (MW)",ticklen= 5,zeroline= False)
             )
fig = go.Figure(data = data, layout = layout)
#url = py.plot(fig, validate=False)
fig.show()


# On constate que de 2008 à 2018, c'est le parc nucléaire qui produit le plus de quantité d'électricité (environ 63k MW) en France, et cela de manière stable. En 2ème position vient le parc hydraulique (25k MW environ) et en 3ème position le parc thermique fossile. La production de cette denière varie de 18 à 27k MW. <br>
# Le parc bioénergie est celui qui produit le moins d'énergie en France. <br>
# Les parcs gaz, fioul, charbon ont produit de l'électricité de manière discontinue (pas de production de 2010 à 2012, puis de 2016 à 2018).

# In[23]:


#Line Plot
trace1 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc eolien (MW)'],
                mode = "lines+markers",
                name = "Parc éolien (MW)",
                marker = dict(color = 'rgb(79, 194, 102)'),
                text = ['Année, Parc éolien (MW)'])
trace2 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc solaire (MW)'],
                mode = "lines+markers",
                name = "Parc solaire (MW)",
                marker = dict(color = 'rgb(0, 247, 255)'),
                text = ['Année, Parc solaire (MW)'])
trace3 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc bioenergie (MW)'],
                mode = "lines+markers",
                name = "Parc bioenergie (MW)",
                marker = dict(color = 'rgb(230, 0, 255)'),
                text = ['Année, Parc bioenergie (MW)'])
trace4 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc hydraulique (MW)'],
                mode = "lines+markers",
                name = "Parc hydraulique (MW)",
                marker = dict(color = 'rgb(250, 236, 80)'),
                text = ['Année, Parc hydraulique (MW)'])
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(title = "Evolution de la quantité d'énergies renouvelables produite par parc en France (2008 à 2018)",
              xaxis= dict(title= 'Année',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Quantité énergétique produite par parc (MW)',ticklen= 5,zeroline= False)
             )
fig = go.Figure(data = data, layout = layout)
#url = py.plot(fig, validate=False)
fig.show()


# Entre 2008 et 2018, au niveau de la production d'électricité provenant de sources renouvelables, c'est le parc hydraulique qui domine largement (production aux alentours de 25k MW). Vient ensuite le parc éolien (quantité d'électricité produite variant de 2k à 16k MW). Le parc solaire vient en 3ème position pour une quantité d'électricité produite variant de 7 à 9k MW. Le parc bioénergie est le parc qui produit le moins d'électricité (aux alentours de 1k MW)

# In[24]:


#Line Plot
trace1 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc gaz (MW)'],
                mode = "lines+markers",
                name = "Parc gaz (MW)",
                marker = dict(color = 'rgb(255, 0, 140)'),
                text = ['Année, Parc gaz (MW)'])
trace2 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc nucleaire (MW)'],
                mode = "lines+markers",
                name = "Parc nucleaire (MW)",
                marker = dict(color = 'rgb(9, 219, 104)'),
                text = ['Année, Parc nucleaire (MW)'])
trace3 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc thermique fossile (MW)'],
                mode = "lines+markers",
                name = "Parc thermique fossile (MW)",
                marker = dict(color = 'rgb(255, 247, 0)'),
                text = ['Année, Parc thermique fossile (MW)'])
trace4 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc fioul (MW)'],
                mode = "lines+markers",
                name = "Parc fioul (MW)",
                marker = dict(color = 'rgb(0, 162, 255)'),
                text = ['Année, Parc fioul (MW)'])
trace5 = go.Scatter(
                x = parc.Annee.sort_values(ascending=False),
                y = parc['Parc charbon (MW)'],
                mode = "lines+markers",
                name = "Parc charbon (MW)",
                marker = dict(color = 'rgb(255, 170, 0)'),
                text = ['Année, Parc charbon (MW)'])
data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(title = "Evolution de la quantité d'energies non-renouvelables produite par parc en France (2008 à 2018)",
              xaxis= dict(title= 'Année',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Quantité énergétique produite par parc (MW)',ticklen= 5,zeroline= False)
             )
fig = go.Figure(data = data, layout = layout)
#url = py.plot(fig, validate=False)
fig.show()


# Au niveau des énergies non-renouvelables produites, c'est le parc nucléaire qui domine largement (production aux alentours de 25k MW). Vient ensuite le parc thermique fossile (quantité d'électricité produite variant de 18k à 17k MW). Le parc gaz vient en 3ème position pour une quantité d'électricité produite se situant aux alentours de 12k MW. Le parc charbon est le parc qui produit le moins d'électricité (variant de 2k à 7k MW). <br>
# On constate par ailleurs pour les 3 parcs suivants : charbon, fioul, gaz, une discontinuité de la production respectivement de 2010 à 2012, puis de 2016 à 2018.

# # Question 2 – Création de profils de production renouvelable

# Les énergies renouvelables éoliennes et solaires ne sont pas pilotables. Afin de déterminer la manière dont ces actifs produisent de l’électricité dans la journée, il est nécessaire de déterminer leur profil de production, qui sera appliqué aux capacités installées dans le mix électrique.
# 

# La puissance électrique installée représente la capacité de production électrique d'un équipement. Le plus souvent exprimée en Mégawatts, voire en Gigawatts, elle peut être d’origine hydraulique, nucléaire, thermique, solaire ou éolienne. (Définition : https://www.planete-energies.com/fr/content/puissance-installee)

# # Question 2.1 - Récupération des données de production

# 1- Dans la base Eco2Mix, récupérons les données de production d’électricité éolienne et solaire de 2012. Dans la base du parc national annuel, récupérons les capacités installées au 31 décembre 2011.

# # a) Récupération des données de production d’électricité éolienne et solaire de 2012

# Récupération des données pour l'année 2012

# In[41]:


date = mix['Date']
heure = mix['Heure']
date_et_heure = mix['Date et Heure']
conso = mix['Consommation (MW)']
eolien = mix['Eolien (MW)']
solaire = mix['Solaire (MW)']

Date = []
Heure = []
Date_et_heure = []
Consommation = []
Eolien = []
Solaire = []

for i in range(0,mix.shape[0]) :
    if(('2012' in date[i])==True):
        #print(i)
        Date.append(date[i])
        Heure.append(heure[i])
        Date_et_heure.append(date_et_heure[i])
        Consommation.append(conso[i])
        Eolien.append(eolien[i])
        Solaire.append(solaire[i])
        
dfdate = pd.DataFrame(Date)
dfheure = pd.DataFrame(Heure)
dfdateheure = pd.DataFrame(Date_et_heure)
dfconso = pd.DataFrame(Consommation)
dfeolien = pd.DataFrame(Eolien)
dfsolaire = pd.DataFrame(Solaire)

#frames = [dfdate, dfheure, dfdateheure, dfconso, df_proportion.Proportion, dfeolien, dfsolaire]
frames = [dfdate, dfheure, dfdateheure, dfconso, dfeolien, dfsolaire]

df3 = pd.concat(frames, axis=1)
df3.columns = ['Date', 'Heure', 'Date_et_heure','Consommation','Eolien','Solaire']
df3 = df3.dropna()
df3 = df3.sort_values(by=['Date_et_heure'])
print(df3.shape)
df3.head()


# Moyenne des productions éoliennes et solaires pour avoir les productions par heure.

# In[169]:


i = 0
k = 0

Date = []
Heure = []
Date_et_heure = []
Consommation = []
Eolien = []
Solaire = []

while i<df3.shape[0]/2:
    #print(i)
    #print(k)
    date = df3.iloc[k,0]
    heure = df3.iloc[k,1]
    date_et_heure = df3.iloc[k,2]
    consommation = df3.iloc[k,3] + df3.iloc[(k+1),3]
    eolien = (df3.iloc[k,4] + df3.iloc[(k+1),4])/2 #divisé par 2, moyenne
    solaire = (df3.iloc[k,5] + df3.iloc[(k+1),5])/2 #divisé par 2, moyenne

    Date.append(date)
    Heure.append(heure)
    Date_et_heure.append(date_et_heure)
    Consommation.append(consommation)
    Eolien.append(eolien)
    Solaire.append(solaire)
    
    k = k+2
    i = i+1

dfDate = pd.DataFrame(Date)
dfHeure = pd.DataFrame(Heure)
dfDate_et_heure = pd.DataFrame(Date_et_heure)
dfConsommation = pd.DataFrame(Consommation)
dfEolien = pd.DataFrame(Eolien)
dfSolaire = pd.DataFrame(Solaire)

frames = [dfDate, dfHeure, dfDate_et_heure, dfConsommation, df_proportion.Proportion, df_profil.Profil_TWh, dfEolien,dfSolaire]
df_eolien_solaire = pd.concat(frames, axis = 1)
df_eolien_solaire.columns = ['Date','Heure','Date_et_heure','Consommation','Profil', 'Consommation_600','Eolien','Solaire']
print("Le dataframe eolien_solaire_par_heure est de taille ",df_eolien_solaire.shape)
df_eolien_solaire.head()


# # b) Récupération des capacités installées au 31 décembre 2011 dans la base du parc national
# 

# In[170]:


p = parc[parc.Annee==2011]
annee_parc = p['Annee']
eolien_parc = p['Parc eolien (MW)']
solaire_parc = p['Parc solaire (MW)']
annee_parc = pd.DataFrame(annee_parc)
eolien_parc = pd.DataFrame(eolien_parc)
solaire_parc = pd.DataFrame(solaire_parc)
#print(dfannee_parc.shape)
#print(dfeolien_parc.shape)
#print(solaire_parc.shape)
frames = [annee_parc,eolien_parc,solaire_parc]
parc_capa_2011 = pd.concat(frames, axis=1)
print(parc_capa_2011.shape)
parc_capa_2011.head()


# # Question 2.2 - Calcul de proportion de production

# Pour chaque heure de 2012, calculons la proportion de la production par rapport à la capacité installée. Observons ensuite la situation ces profils par rapport au profil de consommation ?

# In[222]:


d = df_eolien_solaire.copy()
Ne = parc_capa_2011['Parc eolien (MW)'] #capacité installée en 2011 pour honorer la demande en 2012
Ns = parc_capa_2011['Parc solaire (MW)'] #capacité installée en 2011 pour honorer la demande en 2012

Proportion_Eolien = []
Proportion_Solaire = []


for i in range(0,d.shape[0]):
    proportion_eolien = d.Eolien[i]/int(Ne)
    proportion_solaire = d.Solaire[i]/int(Ns)
    #var_arr = round(var,4) #limiter à 4 chiffres après la virgule
    Proportion_Eolien.append(proportion_eolien)
    Proportion_Solaire.append(proportion_solaire)

dfProportion_Eolien = pd.DataFrame(Proportion_Eolien)
dfProportion_Solaire = pd.DataFrame(Proportion_Solaire)

#dfProportion_Eolien.reset_index(drop=True, inplace=True)
#dfProportion_Solaire.reset_index(drop=True, inplace=True)

frames = [dfProportion_Eolien, dfProportion_Solaire]
dfer = pd.concat(frames, axis=1)
dfer.columns = ['Profil_Eolien','Profil_Solaire'] 
print("Le dataframe dfer est de taille ",dfer.shape)
dfer.head()


# In[266]:


frames = [df_eolien_solaire,dfer]
df_prop_eolien_solaire = pd.concat(frames, axis=1)
df_prop_eolien_solaire.head()


# In[394]:


moy_eol = np.mean(df_prop_eolien_solaire.Profil_Eolien)*100
moy_sol = np.mean(df_prop_eolien_solaire.Profil_Solaire)*100
moy_con = np.mean(df_prop_eolien_solaire.Profil)
print(moy_eol)
print(moy_sol)
print(moy_con)


# La moyenne de la production éolienne est de 25%. <br>
# La moyenne de la production solaire est de 17%. <br>
# La moyenne de la consommation est de 0.011%. <br>
# On peut déduire que les parcs éoliens et solaires produisent plus que ce qui est absorbé par la consommation moyenne annuelle en 2012. Ces parcs sont en surcapacité si on compare la production en 2011 et la consommation en 2012. On peut constater cela rapidement à travers le graphique ci-dessous où l'on voit que ces 3 variables ne sont pas du tout de même ordre de grandeur.

# In[391]:


import plotly.graph_objects as go
animals=['Profil Eolien (moyenne en % pour 2012)', 'Profil Solaire (moyenne en % pour 2012)', 'Consommation (moyenne en % pour 2012)']

fig = go.Figure([go.Bar(x=animals, 
                        y=[moy_eol, moy_sol, moy_con]),
                ])

#data = [trace1, trace2, trace3, trace4, trace5]
#layout = go.Layout(title = "Evolution de profils de production par rapport à la consommation en France (2012)",
              #xaxis= dict(title= 'Profils',ticklen= 5,zeroline= False),
              #yaxis= dict(title= 'Moyenne',ticklen= 5,zeroline= False)
             #)
#fig = go.Figure(data = data, layout = layout)
fig.show()


# # Question 3 - Détermination d'un mix optimal pour 2050

# Afin de constituer le système électrique permettant de servir la consommation prévue en 2050, il faut déterminer les capacités de production (en MW) et la production horaire (en MWh) nécessaires pour satisfaire la demande de chaque heure de l’année au meilleur coût.

# Les coûts des différents actifs sont indiqués dans le tableau ci-dessous : 
#     

# In[146]:


from IPython.display import display, HTML

ar = np.array([
    ["Eolien",70000, 0], 
    ["Solaire",50000, 0], 
    ["CCGT",60000, 150],
    ["Déléstage",0, 3000],
    ["Excès de production",0, 1000]])

dfsynthese = pd.DataFrame(ar, index = ['', '', '','',''], 
                          columns = ['Actif',"Coût d'investissement (€/MW)", "Coût opérationnel (€/MWh)"])

display(dfsynthese)


# Ecrivons le programme linéaire associé à ce problème. <br> L’objectif du programme linéaire est de minimiser les coûts d’installation de capacités et de fonctionnement du système pour les trois actifs suivants : éolien, solaire, ccgt. Ces coûts nous sont donnés dans le tableau ci-dessous. Minimiser ces coûts revient à minimiser la quantité de délestage et d'excès de production <br>
# La demande de chaque heure de la journée doit être satisfaite par la production des différents actifs. <br>
# L’équation également doit tenir compte des délestages ou des excès de production éventuels. <br>
# La production de chaque actif ne peut pas dépasser sa capacité installée. Autrement dit : <br>
# <br>
# $$ \mbox{production} ≤ \mbox{capacité installée} $$
# <br>
# Pour les actifs éoliens et solaires, la production horaire est fixée par le produit entre la capacité installée et le profil déterminé dans la question 2. Autrement dit, <br>
# <br>
# $$ \mbox{production horaire éolienne} = \mbox{capacité installée pour l'actif éolien} * \mbox{proportion de l'éolien}$$
# $$ \mbox{production horaire solaire} = \mbox{capacité installée pour l'actif solaire} * \mbox{proportion du solaire} $$
# <br>

# Nous pouvons modéliser ce problème de la manière suivante : 

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000c_1 +50000c_2+60000c_3 + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c_1*profil_1[i] + c_2*profil_2[i] + p_3[i] + d[i] = c_{600}[i]+e[i]\cr
#        p_3[i]≤c_3\cr
#        c_i≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# avec : <br>
# - c_1 : capacité installée pour l'actif éolien
# - c_2 : capacité installée pour l'actif solaire
# - c_3 : capacité installée pour l'actif ccgt
# - p_3 : production horaire de ccgt
# - d : délestage (qui correspond à la différence entre la consommation et la production)
# - e : excédent de production (qui correspond à la différence entre production et la consommation)
# - profil_1 : proportion de l'actif éolien
# - profil_2 : proportion de l'actif solaire

# In[395]:


df_2050 = df_prop_eolien_solaire.copy()
df_2050.shape
df_2050.head()


# In[450]:


#-------------------
#Resolution de P 
#-------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem = pulp.LpProblem("Problème mix énergétique",pulp.LpMinimize)

#Création des variables
c_1 = pulp.LpVariable('capacité éolien', lowBound=0, cat='Continuous')
c_2 = pulp.LpVariable('capacité solaire', lowBound=0, cat='Continuous')
c_3 = pulp.LpVariable('capacité ccgt', lowBound=0, cat='Continuous')

nb_heures = len(df_2050)
p_3 = [pulp.LpVariable(f"production ccgt à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d = [pulp.LpVariable(f"délestage à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e = [pulp.LpVariable(f"excédent à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem += 70000 * c_1 + 50000 * c_2 + 60000 * c_3 + 150 * sum(p_3) + 3000 * sum(d) + 1000 * sum(e), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem += p_3[i] <= c_3
    problem += c_1*df_2050['Profil_Eolien'].iloc[i] + c_2*df_2050['Profil_Solaire'].iloc[i] + p_3[i] + d[i] == df_2050['Consommation_600'].iloc[i] + e[i]

# Création du fichier .lp 
problem.writeLP("problem_normal.lp")

#Résolution du problème
problem.solve()

# Affichage du statut du problème
pulp.LpStatus[problem.status]


#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1, c_2, c_3 et de la fonction objective
print("La valeur de c_1 est :", c_1.varValue, "MW")
print("La valeur de c_2 est :",c_2.varValue, "MW")
print("La valeur de c_3 est :",c_3.varValue, "MW")
print("La valeur de la fonction objective Z est : ",pulp.value(problem.objective), "MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestages est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Question 4 - Détermination d'un mix optimal 

# Le mix électrique calculé dans la question 2 est adapté à une consommation annuelle de 600 TWh. Cependant, il existe une incertitude importante sur ces projections de demande à long terme, et particulièrement dans le cadre de la transition énergétique. En effet, il est difficile d’évaluer quels vecteurs énergétiques et quelles technologies émergeront pour satisfaire les différents usages à cet horizon.

# # Question 4.1 - Simulation du fonctionnement du système électrique pour des consommations annuelles de 500 TWh et 700 TWh.

# En fixant les capacités des actifs à celles obtenues dans la question 2, simulons le fonctionnement du système électrique pour des consommations annuelles de 500 TWh et 700 TWh. Observons si : 
# - le système s’adapte ou non à ces consommations annuelles
# - les délestages et excès de production varient-ils, et si oui pourquoi

# Dans la première simulation (consommation = 500 TWh), le programme linéaire aura la forme suivante :

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000 c1_{500} +50000 c2_{500}+60000 c3_{500} + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c1_{500} *profil_1[i] + c2_{500} *profil_2[i] + p_3[i] + d[i] = c_{500}[i]+e[i]\cr
#        p_3[i]≤ c3_{500}\cr
#        ci_{500} ≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# Dans la deuxième simulation (consommation = 700 TWh), le programme linéaire aura la forme suivante :

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000 c1_{700} +50000 c2_{700} +60000 c3_{700} + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c1_{700} * profil_1[i] + c2_{700} * profil_2[i] + p_3[i] + d[i] = c_{700}[i]+e[i]\cr
#        p_3[i] ≤ c3_{700}\cr
#        ci_{700} ≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# In[400]:


print(df_2050.shape)
df_2050.head()


# In[273]:


Consommation_500 = []
Consommation_700 = []

for i in range(0,df_2050.shape[0]):
    p1 = df_2050.Profil[i]*5000000
    Consommation_500.append(p1)
    
    p2 = df_2050.Profil[i]*7000000
    Consommation_700.append(p2)

dfConsommation_500 = pd.DataFrame(Consommation_500)
dfConsommation_700 = pd.DataFrame(Consommation_700)
dfConsommation_500.columns = ['Consommation_500']
dfConsommation_700.columns = ['Consommation_700']

frames = [df_2050,dfConsommation_500,dfConsommation_700]
df_2050_bis = pd.concat(frames,axis=1)
print("Le dataframe df_profil est de taille ",df_2050_bis.shape)
df_2050_bis.head()


# In[401]:


df_2050 = df_2050_bis
print(df_2050.shape)
df_2050.head()


# # Simulation 1 : Consommation annuelle = 500 TWh

# In[451]:


#---------------------------------------------
#Résolution de P (pour Consommation = 500 TWh)
#---------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_2 = pulp.LpProblem("Problème mix énergétique hypothèse 2 (Consommation = 500 TWh)",pulp.LpMinimize)

#Création des variables
c_1_500 = c_1.varValue #on fixe la valeur de c_1_500
c_2_500 = c_2.varValue #on fixe la valeur de c_2_500
c_3_500 = c_3.varValue #on fixe la valeur de c_3_500

nb_heures = len(df_2050)

p_3_hyp2 = [pulp.LpVariable(f"production ccgt à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d_hyp2 = [pulp.LpVariable(f"délestage à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e_hyp2 = [pulp.LpVariable(f"excédent à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fonction Objective
problem_2 += 70000 * c_1_500 + 50000 * c_2_500 + 60000 * c_3_500 + 150 * sum(p_3_hyp2) + 3000 * sum(d_hyp2) + 1000 * sum(e_hyp2), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_2 += p_3_hyp2[i] <= c_3_500
    problem_2 += c_1_500*df_2050['Profil_Eolien'].iloc[i] + c_2_500*df_2050['Profil_Solaire'].iloc[i] + p_3_hyp2[i] + d_hyp2[i] == df_2050['Consommation_500'].iloc[i] + e_hyp2[i]

# Création du fichier .lp 
problem_2.writeLP("problem2_normal.lp")
    
#Résolution du problème
problem_2.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_2.status])

#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1_500, c_2_500, c_3_500 et de la fonction objective
print("La valeur de c_1_500 est : ", c_1.varValue,"MW")
print("La valeur de c_2_500 est : ",c_2.varValue,"MW")
print("La valeur de c_3_500 est :",c_3.varValue,"MW")
print("La valeur de la fonction objective Z est :",pulp.value(problem_2.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_2.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_2.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Simulation 2 : Consommation annuelle = 700 TWh

# In[452]:


#---------------------------------------------
#Résolution de P (pour Consommation = 700 TWh)
#---------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_3 = pulp.LpProblem("Problème mix énergétique hypothèse 3 (Consommation =700 TWh)",pulp.LpMinimize)

#Création des variables
c_1_700 = c_1.varValue
c_2_700 = c_2.varValue
c_3_700 = c_3.varValue

nb_heures = len(df_2050)

p_3_hyp3 = [pulp.LpVariable(f"production ccgt à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d_hyp3 = [pulp.LpVariable(f"délestage à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e_hyp3 = [pulp.LpVariable(f"excédent à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem_3 += 70000 * c_1_700 + 50000 * c_2_700 + 60000 * c_3_700 + 150 * sum(p_3_hyp3) + 3000 * sum(d_hyp3) + 1000 * sum(e_hyp3), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_3 += p_3_hyp3[i] <= c_3_700
    problem_3 += c_1_700*df_2050['Profil_Eolien'].iloc[i] + c_2_700*df_2050['Profil_Solaire'].iloc[i] + p_3_hyp3[i] + d_hyp3[i] == df_2050['Consommation_700'].iloc[i] + e_hyp3[i]
   
 # Création du fichier .lp
problem_3.writeLP("problem3_normal.lp")

#Résolution du problème
problem_3.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_3.status])

#Affichage des valeurs variables
#for variable in problem_3.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1_700, c_2_700, c_3_700 et de la fonction objective
print("La valeur de c_1_700 est : ", c_1.varValue,"MW")
print("La valeur de c_2_700 est : ",c_2.varValue,"MW")
print("La valeur de c_3_700 est : ",c_3.varValue,"MW")
print("La valeur de la fonction objective Z est :",pulp.value(problem_3.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_3.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_3.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Question 5 – Prise en compte de plusieurs scénarios

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000c_1 +50000c_2+60000c_3 + 1/3 * (150\sum_{i=1}^{8784} p3_{500}[i] + 3000\sum_{i=1}^{8784} d_{500}[i] + 1000\sum_{i=1}^{8784} e_{500}[i]) + (1/3) * (150\sum_{i=1}^{8784} p3_{600}[i] + 3000\sum_{i=1}^{8784} d_{600}[i] + 1000\sum_{i=1}^{8784} e_{600}[i]) + (1/3) * (150\sum_{i=1}^{8784} p3_{700}[i] + 3000\sum_{i=1}^{8784} d_{700}[i] + 1000\sum_{i=1}^{8784} e_{700}[i]) \cr
#        c_1*profil_1[i] + c_2*profil_2[i] + p3_{500}[i] + d_{500}[i] = c_{500}[i]+e_{500}[i]\cr
#        c_1*profil_1[i] + c_2*profil_2[i] + p3_{600}[i] + d_{600}[i] = c_{600}[i]+e_{600}[i]\cr
#        c_1*profil_1[i] + c_2*profil_2[i] + p3_{700}[i] + d_{700}[i] = c_{700}[i]+e_{700}[i]\cr
#        p3_{500}[i]≤c_3\cr
#        p3_{600}[i]≤c_3\cr
#        p3_{700}[i]≤c_3\cr
#        c_i≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# In[453]:


#--------------------------------------------------------------------------------
#Résolution de P (pour 3 scenarios : Consommations = 500 TWh, 600 TWh et 700 TWh)
#--------------------------------------------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_4 = pulp.LpProblem("Problème mix énergétique hypothèse 4 (prise en compte de plusieurs scenarios)",pulp.LpMinimize)

#Création des variables
c_1 = pulp.LpVariable('capacité éolien', lowBound=0, cat='Continuous')
c_2 = pulp.LpVariable('capacité solaire', lowBound=0, cat='Continuous')
c_3 = pulp.LpVariable('capacité ccgt', lowBound=0, cat='Continuous')


nb_heures = len(df_2050)

p_3_500 = [pulp.LpVariable(f"production ccgt à l'heure {i} hypothèse 1", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d_500 = [pulp.LpVariable(f"délestage à l'heure {i} hypothèse 1", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e_500 = [pulp.LpVariable(f"excédent à l'heure {i} hypothèse 1", lowBound=0, cat='Continuous') for i in range(nb_heures)]

p_3_600 = [pulp.LpVariable(f"production ccgt à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d_600 = [pulp.LpVariable(f"délestage à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e_600 = [pulp.LpVariable(f"excédent à l'heure {i} hypothèse 2", lowBound=0, cat='Continuous') for i in range(nb_heures)]

p_3_700 = [pulp.LpVariable(f"production ccgt à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d_700 = [pulp.LpVariable(f"délestage à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e_700 = [pulp.LpVariable(f"excédent à l'heure {i} hypothèse 3", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem_4 += 70000 * c_1 + 50000 * c_2 + 60000 * c_3 + 1/3*(150 * sum(p_3_500) + 3000 * sum(d_500) + 1000 * sum(e_500)) + 1/3*(150 * sum(p_3_600) + 3000 * sum(d_600) + 1000 * sum(e_600)) + 1/3*(150 * sum(p_3_700) + 3000 * sum(d_700) + 1000 * sum(e_700)), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_4 += p_3_500[i] <= c_3
    problem_4 += p_3_600[i] <= c_3
    problem_4 += p_3_700[i] <= c_3

    problem_4 += c_1*df_2050['Profil_Eolien'].iloc[i] + c_2*df_2050['Profil_Solaire'].iloc[i] + p_3_500[i] + d_500[i] == df_2050['Consommation_500'].iloc[i] + e_500[i]
    problem_4 += c_1*df_2050['Profil_Eolien'].iloc[i] + c_2*df_2050['Profil_Solaire'].iloc[i] + p_3_600[i] + d_600[i] == df_2050['Consommation_600'].iloc[i] + e_600[i]
    problem_4 += c_1*df_2050['Profil_Eolien'].iloc[i] + c_2*df_2050['Profil_Solaire'].iloc[i] + p_3_700[i] + d_700[i] == df_2050['Consommation_700'].iloc[i] + e_700[i]


# Création du fichier .lp
problem_4.writeLP("problem4_normal.lp")

#Résolution du problème
problem_4.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_4.status])

#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")

# Affichage de la valeur de c_1, c_2, c_3 et de la fonction objective
print("La nouvelle valeur de c_1 est : ", c_1.varValue,"MW")
print("La nouvelle valeur de c_2 est : ",c_2.varValue,"MW")
print("La nouvelle valeur de c_3 est : ",c_3.varValue,"MW")
print("La valeur de la fonction objective Z est :",pulp.value(problem_3.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_4.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_4.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité total d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Question 6 : prise en compte des résultats des scenarios (nouvelles valeurs de c_1, c_2 et c_3)

# # Cas 1 : Consommation = 500 TWh

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000c_1{_n} +50000c_2{_n}+60000c_3{_n} + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c_1{_n}*profil_1[i] + c_2{_n}*profil_2[i] + p_3[i] + d[i] = c_{500}[i]+e[i]\cr
#        p_3[i]≤c_3{_n}\cr
#        c_i{_n}≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# In[454]:


#-------------------------------------------------------------------------------------
#Resolution de P (Consommation = 500 TWh avec les valeurs c1, c2, c3 de la question 5)
#-------------------------------------------------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_5 = pulp.LpProblem("Problème mix énergétique (Consommation = 500 TWh) et nouvelles valeurs de c_1, c_2 et c_3",pulp.LpMinimize)

#Création des variables
c_1_n = c_1.varValue
c_2_n = c_2.varValue
c_3_n = c_3.varValue

nb_heures = len(df_2050)

p_3 = [pulp.LpVariable(f"production ccgt à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d = [pulp.LpVariable(f"délestage à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e = [pulp.LpVariable(f"excédent à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem_5 += 70000 * c_1_n + 50000 * c_2_n + 60000 * c_3_n + 150 * sum(p_3) + 3000 * sum(d) + 1000 * sum(e), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_5 += p_3[i] <= c_3_n
    problem_5 += c_1_n*df_2050['Profil_Eolien'].iloc[i] + c_2_n*df_2050['Profil_Solaire'].iloc[i] + p_3[i] + d[i] == df_2050['Consommation_500'].iloc[i] + e[i]

    
# Création du fichier .lp
problem_5.writeLP("problem5_normal.lp")

#Résolution du problème
problem_5.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_5.status])

#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1, c_2, c_3 et de la fonction objective
print("La nouvelle valeur de c_1_n est : ", c_1.varValue,"MW")
print("La nouvelle valeur de c_2_n est : ",c_2.varValue,"MW")
print("La nouvelle valeur de c_3_n est : ",c_3.varValue,"MW")
print("La valeur de la fonction objective est : ",pulp.value(problem_5.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_5.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_5.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Cas 2 : Consommation = 600 TWh

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000c_1{_n} +50000c_2{_n}+60000c_3{_n} + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c_1{_n}*profil_1[i] + c_2{_n}*profil_2[i] + p_3[i] + d[i] = c_{600}[i]+e[i]\cr
#        p_3[i]≤c_3{_n}\cr
#        c_i{_n}≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# In[455]:


#-------------------------------------------------------------------------------------
#Resolution de P (Consommation = 600 TWh avec les valeurs c1, c2, c3 de la question 5)
#-------------------------------------------------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_6 = pulp.LpProblem("Problème mix énergétique (Consommation = 600 TWh) et nouvelles valeurs de c_1, c_2 et c_3",pulp.LpMinimize)

#Création des variables
c_1_n = c_1.varValue
c_2_n = c_2.varValue
c_3_n = c_3.varValue

nb_heures = len(df_2050)

p_3 = [pulp.LpVariable(f"production ccgt à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d = [pulp.LpVariable(f"délestage à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e = [pulp.LpVariable(f"excédent à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem_6 += 70000 * c_1_n + 50000 * c_2_n + 60000 * c_3_n + 150 * sum(p_3) + 3000 * sum(d) + 1000 * sum(e), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_6 += p_3[i] <= c_3_n
    problem_6 += c_1_n*df_2050['Profil_Eolien'].iloc[i] + c_2_n*df_2050['Profil_Solaire'].iloc[i] + p_3[i] + d[i] == df_2050['Consommation_600'].iloc[i] + e[i]

    
# Création du fichier .lp
problem_6.writeLP("problem6_normal.lp")

#Résolution du problème
problem_6.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_6.status])

#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1, c_2, c_3 et de la fonction objective
print("La nouvelle valeur de c_1_n est : ", c_1.varValue,"MW")
print("La nouvelle valeur de c_2_n est : ",c_2.varValue,"MW")
print("La nouvelle valeur de c_3_n est : ",c_3.varValue,"MW")
print("La valeur de la fonction objective est : ",pulp.value(problem_6.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_6.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_6.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)


# # Cas 3 : Consommation = 700 TWh

# $$
# (P):
# \left \{
# \begin{array}{r c l}
# Min \: Z = 70000c_1{_n} +50000c_2{_n}+60000c_3{_n} + 150\sum_{i=1}^{8784} p_3[i] + 3000\sum_{i=1}^{8784} d[i] + 1000\sum_{i=1}^{8784} e[i]\cr
#        c_1{_n}*profil_1[i] + c_2{_n}*profil_2[i] + p_3[i] + d[i] = c_{700}[i]+e[i]\cr
#        p_3[i]≤c_3{_n}\cr
#        c_i{_n}≥0 \: ∀ i∈\{1,...,8784\} \cr
# \end{array}
# \right .
# $$

# In[456]:


#-------------------------------------------------------------------------------------
#Resolution de P (Consommation = 700 TWh avec les valeurs c1, c2, c3 de la question 5)
#-------------------------------------------------------------------------------------

import pulp #si pas installé, faire !pip install pulp

# df_2050 = mon data frame

problem_7 = pulp.LpProblem("Problème mix énergétique (Consommation = 700 TWh) et nouvelles valeurs de c_1, c_2 et c_3",pulp.LpMinimize)

#Création des variables
c_1_n = c_1.varValue
c_2_n = c_2.varValue
c_3_n = c_3.varValue

nb_heures = len(df_2050)

p_3 = [pulp.LpVariable(f"production ccgt à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
d = [pulp.LpVariable(f"délestage à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]
e = [pulp.LpVariable(f"excédent à l'heure {i}", lowBound=0, cat='Continuous') for i in range(nb_heures)]

# Fontion Objective
problem_7 += 70000 * c_1_n + 50000 * c_2_n + 60000 * c_3_n + 150 * sum(p_3) + 3000 * sum(d) + 1000 * sum(e), "Z"

# Contraintes
for i in range(len(df_2050)):
    problem_7 += p_3[i] <= c_3_n
    problem_7 += c_1_n*df_2050['Profil_Eolien'].iloc[i] + c_2_n*df_2050['Profil_Solaire'].iloc[i] + p_3[i] + d[i] == df_2050['Consommation_700'].iloc[i] + e[i]

    
# Création du fichier .lp
problem_7.writeLP("problem7_normal.lp")

#Résolution du problème
problem_7.solve()

# Affichage du statut du problème
print("Le statut du problème est : ",pulp.LpStatus[problem_7.status])

#Affichage des valeurs variables
#for variable in problem.variables():
    #print(f"{variable.name} = {variable.varValue}")
    
# Affichage de la valeur de c_1, c_2, c_3 et de la fonction objective
print("La nouvelle valeur de c_1_n est : ", c_1.varValue,"MW")
print("La nouvelle valeur de c_2_n est : ",c_2.varValue,"MW")
print("La nouvelle valeur de c_3_n est : ",c_3.varValue,"MW")
print("La valeur de la fonction objective est : ",pulp.value(problem_7.objective),"MW")

heures_delestage = 0
volume_delestage = 0
for variable in problem_7.variables():
    if ("délestage" in variable.name) & (variable.varValue != 0):
        heures_delestage+=1
        volume_delestage+=variable.varValue
        
print("Le nombre d'heures de délestage est égal à : %d" %heures_delestage)
print("La quantité totale de délestage est égale à : %.3f MW" %volume_delestage)
        
heures_excedent = 0
volume_excedent = 0
for variable in problem_7.variables():
    if ("excédent" in variable.name) & (variable.varValue != 0):
        heures_excedent+=1
        volume_excedent+=variable.varValue
        
print("Le nombre d'heures d'excès de production est égal à : %d" %heures_excedent)
print("La quantité totale d'excès de production est égale à : %.3f MW" %volume_excedent)

