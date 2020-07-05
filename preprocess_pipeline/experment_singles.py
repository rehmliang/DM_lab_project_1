from preprocess_pipeline import singles

trainingData, testData = singles.load()

print('dataset: singles')
print('protected index: %s, protected value %s' % (singles.protectedIndex, singles.protectedValue))
print('training set: %i' % len(trainingData))
print('test set: %i' % len(testData))
print(trainingData[1])

