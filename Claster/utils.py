import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def load_embeddings(txt_file, max_rows_to_load=None):
    embeddings = np.loadtxt(txt_file, delimiter=',', max_rows=max_rows_to_load)
    return embeddings

def pca(embeddings):
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    df_pca = pca.transform(embeddings)
    df_pca = pd.DataFrame(df_pca, columns=['P1', 'P2'])
    return df_pca

def plot_pca(df_pca):
    fig = px.scatter(df_pca, x='P1', y='P2', title='PCA')
    fig.update_xaxes(title_text='PCA 1')
    fig.update_yaxes(title_text='PCA 2')
    fig.show()

def plot_dendograms(df_pca):
    plt.figure(figsize = (10,7))
    plt.title('Dendrograms')
    plt.axhline(y=3, color='r', linestyle='--')
    dend = sch.dendrogram(sch.linkage(df_pca, method='ward'))
    return dend



"""def agg_clustering(df_pca, n_clusters):
    for n in range(2, n_clusters+1):
        model = AgglomerativeClustering(n_clusters=n)
        y_means = model.fit_predict(df_pca)
        return y_means

def calculate_scores(df_pca, y_means):
    silhouette = silhouette_score(df_pca, y_means)
    print('Silhouette Score: ', silhouette)

    davies_bouldin = davies_bouldin_score(df_pca, y_means)
    print('Davies Bouldin Score: ', davies_bouldin)

def plot_clustering(df_pca, y_means):
    fig = px.scatter(df_pca, x='P1', y='P2', color=y_means, title='Agglomerative Clustering with 4 Clusters', 
                     labels={'color': 'Cluster'})
    fig.update_xaxes(title_text='PCA 1')
    fig.update_yaxes(title_text='PCA 2')
    fig.show()"""

def agg_clustering_with_scores(df_pca, n_clusters):
    fig = go.Figure()

    for n in range(2, n_clusters + 1):
        model = AgglomerativeClustering(n_clusters=n)
        y_means = model.fit_predict(df_pca)

        # Calculate scores
        silhouette = silhouette_score(df_pca, y_means)
        davies_bouldin = davies_bouldin_score(df_pca, y_means)

        # Add scatter plot to the figure
        fig.add_trace(go.Scatter(x=df_pca['P1'], y=df_pca['P2'], mode='markers', marker=dict(color=y_means),
                                 name=f'Clusters: {n}\nSilhouette: {silhouette:.2f}, Davies Bouldin: {davies_bouldin:.2f}'))

    # Update layout
    fig.update_layout(title_text='Agglomerative Clustering with Scores',
                      xaxis=dict(title='PCA 1'), yaxis=dict(title='PCA 2'))

    fig.show()

def cluster_cross_table(df_pca1, df_pca2, n_clusters):
    # Perform clustering on both datasets
    clusters1 = AgglomerativeClustering(n_clusters).fit_predict(df_pca1)
    clusters2 = AgglomerativeClustering(n_clusters).fit_predict(df_pca2)

    # Create a DataFrame for cross-cluster table
    cross_table = pd.crosstab(clusters1, clusters2, rownames=['Clusters in Embeddings 1'], colnames=['Clusters in Embeddings 2'])

    return cross_table

def cluster_cross_table_and_plot(df_pca1, df_pca2, n_clusters_range=(2, 5)):
    # Ensure that both datasets have the same number of rows
    min_rows = min(df_pca1.shape[0], df_pca2.shape[0])
    df_pca1 = df_pca1.iloc[:min_rows, :]
    df_pca2 = df_pca2.iloc[:min_rows, :]

    # Create subplots
    num_subplots = n_clusters_range[1] - n_clusters_range[0] + 1

    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        # Perform clustering on both datasets
        clusters1 = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(df_pca1)
        clusters2 = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(df_pca2)

        # Create a DataFrame for cross-cluster table
        cross_table = pd.crosstab(clusters1, clusters2, rownames=['Clusters in Embeddings 1'], colnames=['Clusters in Embeddings 2'])

        # Create confusion matrix
        confusion_mat = confusion_matrix(clusters1, clusters2)

        # Create heatmap using Plotly Express
        fig = px.imshow(confusion_mat,
                        labels=dict(x='Clusters in Embeddings 2', y='Clusters in Embeddings 1'),
                        x=['Cluster {}'.format(i) for i in range(n_clusters)],
                        y=['Cluster {}'.format(i) for i in range(n_clusters)],
                        color_continuous_scale="Blues",
                        title='{} Clusters'.format(n_clusters))

        # Show the plot
        fig.show()