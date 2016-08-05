import tweepy
import csv
import pandas as pd
from pattern.en import parsetree
from collections import defaultdict
import difflib
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

##################
## ReadIn Tweets
##################
## Create a 2D dictionary
## Assuming tweets are collected and saved in csv format from TweetClassification.py in the same repo. 
Adele = defaultdict(dict)
with open('/PATH/TO/READ/Adele_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f, delimiter=',')
    for index,row in enumerate(reader):
        if (index <100):
            Adele[index]["title"] = row[2]

Clinton = defaultdict(dict)
with open('/PATH/TO/READ/billclinton_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for index,row in enumerate(reader):
        if (index <100):
            Clinton[index]["title"] = row[2]

Trump = defaultdict(dict)
with open('/PATH/TO/READ/realDonaldTrump_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for index,row in enumerate(reader):
        if (index <200):
            Trump[index]["title"] = row[2]

##################
## Visualisation
##################

## Reproducing from:
## http://nbviewer.jupyter.org/github/AYLIEN/headline_analysis/blob/06f1223012d285412a650c201a19a1c95859dca1/main-chunks.ipynb#A-primer-on-parse-trees

for i in range(len(Adele)):
	Adele[i]['title_length'] = len(Adele[i]['title'])
	Adele[i]['title_chunks'] = [chunk.type for chunk in parsetree(Adele[i]['title'])[0].chunks]
	Adele[i]['title_chunks_length'] = len(Adele[i]['title_chunks'])

for i in range(len(Clinton)):
    	Clinton[i]["title_length"] = len(Clinton[i]["title"])
    	Clinton[i]["title_chunks"] = [chunk.type for chunk in parsetree(Clinton[i]["title"])[0].chunks]
    	Clinton[i]["title_chunks_length"] = len(Clinton[i]["title_chunks"])

for i in range(len(Trump)):
    	Trump[i]["title_length"] = len(Trump[i]["title"])
    	Trump[i]["title_chunks"] = [chunk.type for chunk in parsetree(Trump[i]["title"])[0].chunks]
    	Trump[i]["title_chunks_length"] = len(Trump[i]["title_chunks"])

df1 = pd.DataFrame.from_dict(Adele)
df2 = pd.DataFrame.from_dict(Clinton)
df3 = pd.DataFrame.from_dict(Trump)

chunks_joint = []
for i in range (len(Adele)):
	chunks_joint.append(Adele[i]["title_chunks"])
for i in range (len(Clinton)):
	chunks_joint.append(Clinton[i]["title_chunks"])
for i in range (len(Trump)):
	chunks_joint.append(Trump[i]["title_chunks"])

titles_joint = []
for i in range (len(Adele)):
	titles_joint.append(Adele[i]["title"])
for i in range (len(Clinton)):
	titles_joint.append(Clinton[i]["title"])
for i in range (len(Trump)):
	titles_joint.append(Trump[i]["title"])

m_joint = np.zeros((400,400))

for i, chunkx in enumerate(chunks_joint):
    for j, chunky in enumerate(chunks_joint):
        sm=difflib.SequenceMatcher(None,chunkx,chunky)
        m_joint[i][j] = sm.ratio()

#from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_joint = tsne_model.fit_transform(m_joint)

#import bokeh.plotting as bp
#from bokeh.models import HoverTool, BoxSelectTool
#from bokeh.plotting import figure, show, output_notebook

plot_joint = bp.figure(plot_width=600, plot_height=600, title="Adele, Clinton, Trump",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

# Customized list of colors to choose from
colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])

plot_joint.scatter(x=tsne_joint[:,0], y=tsne_joint[:,1],
                    color=colormap[([0] * 200 + [3] * 200)],
                    source=bp.ColumnDataSource({
						"chunks": [Adele[x]["title_chunks"] for x in Adele] + [Clinton[x]["title_chunks"] for x in Clinton]+ [Trump[x]["title_chunks"] for x in Trump],
                        "title": ["Adele: "+ Adele[x]['title'] for x in Adele] + ["Bill Clinton: "+ Clinton[x]['title'] for x in Clinton] + ["Trump: " +Trump[x]['title'] for x in Trump]
                    }))

hover = plot_joint.select(dict(type=HoverTool))
hover.tooltips={"title": "@title"}
show(plot_joint)
