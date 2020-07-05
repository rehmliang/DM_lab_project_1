from preprocess_pipeline import adult
import pandas as pd

trainingData, testData = adult.load()

print('dataset: adult')
print('protected index: %s, protected value %s' % (adult.protectedIndex, adult.protectedValue))
print('training set: %i' % len(trainingData))
print('test set: %i' % len(testData))
print(trainingData[1])

#train = pd.DataFrame(german.data)
test = pd.DataFrame(adult.test)

print(test.info())