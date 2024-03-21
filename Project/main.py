import graph
import utils

def main():
    data, label = utils.load_data()
    classes = ["Fake", "Real"]
    
    graph.plot_histogram(data=data, label=label, classes=classes)
    graph.plot_scatter(data=data, label=label, classes=classes)
    
    utils.compute_statistics(data=data)

if __name__ == "__main__":
    main()