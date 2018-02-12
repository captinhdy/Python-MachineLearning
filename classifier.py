from NeuralNetwork import NeuralNetwork
import StockAnalyzer as sa
import WebScraper as ws
from scrapy.crawler import CrawlerProcess 
from CSVReader import CSVReader
from random import random
from random import randrange

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# Load a CSV file

 
if __name__ == '__main__':

    trainingFilename = 'NLPBagofWords.csv'
    testFilename = 'NLPBagofWords.csv'
    reader = CSVReader()
    network = NeuralNetwork()
    MySpider = ws.QuotesSpider()

    training_dataset = reader.load_csv(trainingFilename)
    testing_dataset = reader.load_csv(testFilename)

    for i in range(len(training_dataset[0])-1):
        reader.str_column_to_float(training_dataset,i)

    for i in range(len(testing_dataset[0])-1):
        reader.str_column_to_float(testing_dataset,i)

    reader.str_column_to_int(training_dataset,len(training_dataset[0])-1)
    reader.str_column_to_int(testing_dataset, len(testing_dataset[0])-1)

    #training_minmax = network.dataset_minmax(training_dataset)
    #network.normalize_dataset(training_dataset, training_minmax)

    #testing_minmax = network.dataset_minmax(testing_dataset)
    #network.normalize_dataset(testing_dataset, testing_minmax)

    l_rate = 0.3
    n_epoch = len(training_dataset) -1
    n_hidden = 5
    n_inputs = len(training_dataset[0])-1
    n_outputs = len(set([row[-1] for row in training_dataset]))

    
    predicted = network.back_propagation(training_dataset, testing_dataset,  l_rate, n_epoch, n_hidden)
    actual = [row[-1] for row in testing_dataset]
    accuracy = network.accuracy_metric(actual, predicted)
    print('Mean Accuracy: %.3f%%' % accuracy)

    #exit = False
    #while exit != True:
    
    #    septal_length = raw_input('Iris septal length: ')
    #    septal_width = raw_input('Iris septal width: ')
    #    pedal_length = raw_input('Iris pedal length: ')
    #    pedal_width = raw_input('Iris pedal width: ')

    data = [float(1), float(1), float(0), None]

    network.predict(data)
    data = [float(1), float(0), float(1), None]

    network.predict(data)
    
    #    cont = raw_input('Do you want to quit y/n')

    #    if cont == 'y':
    #        exit = True


   
    #process = CrawlerProcess({'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'})

    #process.crawl(MySpider)
    #process.start()
    
    #stockAnalyzer = sa.StockAnalyzer();
    #stockAnalyzer.GetIExtradingStockInfo("MSFT")