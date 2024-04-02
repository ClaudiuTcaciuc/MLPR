import graph
import utils

def main():
    data, label = utils.load_data()
    
    classes = {
        "Fake": "blue",
        "Real": "orange"
    }
    
    graph.plot_histogram(data=data, label=label, classes=classes)
    graph.plot_scatter(data=data, label=label, classes=classes)
    graph.plot_correlation_matrix(data=data, label=label)
    graph.plot_pca_explained_variance(data=data)
    graph.plot_lda_histogram(data=data, label=label, classes=classes)

if __name__ == "__main__":
    main()