from preprocess_pipeline import german

trainingData, testData = german.load()

print('dataset: german')
print('protected index: %s, protected value %s' % (german.protectedIndex, german.protectedValue))
print('training set: %i' % len(trainingData))
print('test set: %i' % len(testData))
print(trainingData[1])

def precomputedLabelStatisticalParity(data, labels, protectedIndex, protectedValue, weights=None):
   if weights is None:
      weights = [1] * len(data)

   protectedClass = [(x, wt, l) for (x, wt, l) in zip(data, weights, labels)
                     if x[protectedIndex] == protectedValue]
   elseClass = [(x, wt, l) for (x, wt, l) in zip(data, weights, labels)
                if x[protectedIndex] != protectedValue]

   if len(protectedClass) == 0:
      print("Nobody in the protected class")
      return sum(w for (x, w, l) in elseClass if l == 1) / sum(w for (x, w, l) in elseClass)
   elif len(elseClass) == 0:
      print("Nobody in the else class")
      return -sum(w for (x, w, l) in protectedClass if l == 1) / sum(w for (x, w, l) in protectedClass)
   else:
      protectedProb = sum(w for (x, w, l) in protectedClass if l == 1) / sum(w for (x, w, l) in protectedClass)
      elseProb = sum(w for (x, w, l) in elseClass if l == 1) / sum(w for (x, w, l) in elseClass)

   return elseProb - protectedProb

def signedStatisticalParity(data, protectedIndex, protectedValue, h=None, weights=None):
   if len(data[0]) == 2:  # should do better type checking here...
      pts, labels = zip(*data)
   else:
      pts = data
      labels = None

   if h is not None:
      labels = [h(x) for x in pts]

   if labels is None:
      raise Exception("Must provide either labels or a hypothesis to signedStatisticalParity")
   return precomputedLabelStatisticalParity(pts, labels, protectedIndex, protectedValue, weights)

print('statistical parity of training set: %s' % signedStatisticalParity(trainingData,0,0))
print('statistical parity of test set: %s' % signedStatisticalParity(testData,0,0))