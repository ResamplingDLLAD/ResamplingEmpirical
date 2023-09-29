import pickle
import random

import numpy as np
from sklearn.utils import shuffle
from LogADEmpirical.logadempirical.logdeep.dataset.sample import load_features, sliding_window
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

def load(dataset, ratio, method):
    print("Loading vocab ...")
    with open("./results/bgl/100-0.25-SMOTE/logrobust_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    print("vocab Size: ", len(vocab))

    print("Loading train dataset\n")
    data = load_features("./results/bgl/100-0.25-SMOTE/train.pkl", only_normal=False)
    train_logs, train_labels = sliding_window(data,
                                              vocab=vocab,
                                              window_size=10,
                                              data_dir='./dataset/bgl/',
                                              is_predict_logkey=False,
                                              semantics=True,
                                              sample_ratio=1,
                                              e_name='embeddings.json',
                                              in_size=300
                                              )

    train_logs, train_labels = shuffle(train_logs, train_labels)

    sampling_ratio = 0.25
    per = sampling_ratio

    sampling_method = 'N'
    o_ratio = int(train_labels.count(0)) / int(train_labels.count(1))

    if sampling_method == "SMOTE":
        if per != 1:
            sm = SMOTE(random_state=42, sampling_strategy=1 / (o_ratio * per))
        else:
            sm = SMOTE(random_state=42, sampling_strategy=1)
    elif sampling_method == "ADASYN":
        if per != 1:
            sm = ADASYN(random_state=42, sampling_strategy=1 / (o_ratio * per))
        else:
            sm = ADASYN(random_state=42, sampling_strategy=1)
    elif sampling_method == "NearMiss":
        if per != 1:
            sm = NearMiss(sampling_strategy=1 / (o_ratio * per))
        else:
            sm = NearMiss(sampling_strategy=1)
    elif sampling_method == "ClusterCentroids":
        sm = ClusterCentroids(random_state=42, sampling_strategy=1 / (o_ratio * per))
    elif sampling_method == "InstanceHardnessThreshold":
        sm = InstanceHardnessThreshold(random_state=42, sampling_strategy=1 / (o_ratio * per))
    elif sampling_method == "SMOTEENN":
        sm = SMOTEENN(random_state=42, sampling_strategy=1 / (o_ratio * per))
    elif sampling_method == "SMOTETomek":
        if per != 1:
            sm = SMOTETomek(random_state=42, sampling_strategy=1 / (o_ratio * per))
        else:
            sm = SMOTETomek()

    elif sampling_method == "RandomOverSampler":
        sm = RandomOverSampler(random_state=42, sampling_strategy=1 / (o_ratio * per))

    elif sampling_method == "RandomUnderSampler":
        sm = RandomUnderSampler(random_state=42, sampling_strategy=1 / (o_ratio * per))

    elif sampling_method == 'randomO':
        abnormalIndex, normalIndex = [], []
        for i in range(len(train_labels)):
            if train_labels[i] == 1:
                abnormalIndex.append(i)
            else:
                normalIndex.append(i)
        na = int(len(normalIndex) / (o_ratio * per)) - len(abnormalIndex)  # need to add the number of anomaly
        # na = int(len(abnormalIndex) * per)
        if (na > len(abnormalIndex)):
            addIndex = random.choices(abnormalIndex, k=na)
        else:
            addIndex = random.sample(abnormalIndex, na)
        os_train_logs, os_train_labels = [], []
        for i in range(len(train_logs)):
            if i in addIndex:
                os_train_logs.append(train_logs[i])
                os_train_labels.append(train_labels[i])
            else:
                continue
        train_logs = train_logs + os_train_logs
        train_labels = train_labels + os_train_labels

    elif sampling_method == 'randomU':
        normalIndex, abnormalIndex = [], []
        for i in range(len(train_labels)):
            if train_labels[i] == 0:
                normalIndex.append(i)
            else:
                abnormalIndex.append(i)
        n = len(normalIndex) - int(o_ratio * per * len(abnormalIndex))  # the number of removing normal data
        removeIndex = random.sample(normalIndex, n)
        us_train_logs, us_train_labels = [], []
        for i in range(len(train_logs)):
            if i in removeIndex:
                continue
            else:
                us_train_logs.append(train_logs[i])
                us_train_labels.append(train_labels[i])
        train_logs = us_train_logs
        train_labels = us_train_labels

    if sampling_method != 'N' and sampling_method != 'randomU' \
            and sampling_method != 'randomO':
        print("Sampling method: ", sampling_method)
        print("Sampling ratio: ", str(sampling_ratio))
        train_logs, train_labels = sm.fit_resample(train_logs, train_labels)

    with open("./results/sampling/{}_{}_{}.pkl".format(dataset, ratio, method), mode="wb") as f:
        pickle.dump((train_logs, train_labels), f, protocol=pickle.HIGHEST_PROTOCOL)


def draw(method, ratio):
    # Draw the TSNE graph
    dataset = "bgl"
    print('Dataset is loading')
    # with open("./results/sampling/{}_{}_{}.pkl".format(dataset,ratio,method), mode="rb") as f:
    #     train_logs, train_labels = pickle.load(f)
    with open("./results/tsne_emb/{}_{}_{}_label.pkl".format(dataset, ratio, method), mode="rb") as f:
        train_labels = pickle.load(f)
    with open("./results/tsne_emb/{}_{}_{}.pkl".format(dataset, ratio, method), mode="rb") as f2:
        # pickle.dump((tsne_results[:, 0], tsne_results[:, 1]), f2, protocol=pickle.HIGHEST_PROTOCOL)
        x, y = pickle.load(f2)
    print('Dataset is loaded')
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(np.array(train_logs))
    label_y = ["anomaly" if y == 1 else "normal" for y in train_labels]
    # fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=label_y, symbol=label_y)
    x_1 = []
    x_0 = []
    y_1 = []
    y_0 = []
    label_y_1 = []
    label_y_0 = []
    for fet_x, fet_y, label in zip(x, y, train_labels):
        if label == 1:
            x_1.append(fet_x)
            y_1.append(fet_y)
            label_y_1.append('anomaly')
        else:
            x_0.append(fet_x)
            y_0.append(fet_y)
            label_y_0.append('normal')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_0, y=y_0,
                   mode="markers",
                   marker=dict(size=5,
                               color=px.colors.qualitative.Safe[0],
                               symbol='circle',
                               line=dict(width=0.5,
                                         color='DarkSlateGrey'),),
                   name="normal"))
    fig.add_trace(
        go.Scatter(x=x_1, y=y_1,
                   mode="markers",
                   marker=dict(size=5,
                               color=px.colors.qualitative.Safe[1],
                               symbol='diamond',
                               # symbol='circle',
                               line=dict(width=0.5,
                                         color='DarkSlateGrey')),
                   name="abnormal",
                   ))
    fig.update_layout(
        font=dict(
            family="Libertine",
            size=25,
            color="black",
        ),
        plot_bgcolor="#fff",
        # title=dict(text=f"sampling_method = {method}, sampling_ratio={ratio}",
        #            font=dict(size=20), automargin=True, yref='container'),
        xaxis_title='x',
        yaxis_title="y",
        legend_title_text='label',
        legend=dict(
            # yanchor="top",
            # y=0.7,
            # xanchor="right",
            # x=1.3
            orientation="h",
            yanchor = "top",
            y = 1.12,
            xanchor = "left",
            x = 0.02
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=1, gridcolor='black',
                     zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=1, gridcolor='black',
                     zeroline=True, zerolinewidth=1, zerolinecolor='black')
    # fig.show()
    fig.write_image("{}_{}_{}.svg".format(dataset, ratio, method))


if __name__ == "__main__":
    ratio = 0.75
    method = 'InstanceHardnessThreshold'
    dataset = "bgl"
    draw(method, ratio)
