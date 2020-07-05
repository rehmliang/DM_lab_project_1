from preprocess_pipeline.datautils import *

name = "singles"
protectedIndex = 0 # gender
protectedValue = 1 # female

def toInt(val):
   if val.isdigit():
      return int(val)
   else:
      return -1

def processLine(line):
   values = line.strip().split(',')
   (income, sex, marital, age, educ, occup, resid, dualInc, perInHou,
    under18, homeStatus, homeType, ethnic, language) = values

   # shift sex by -1 to get it binary.
   point = [toInt(sex)-1, toInt(ethnic), toInt(age), toInt(educ), toInt(occup),
         toInt(resid), toInt(perInHou), toInt(under18), toInt(homeStatus),
         toInt(homeType), toInt(language)]

   label = 1 if int(income[0]) >= 5 else -1

   return tuple(point), label

def load():
   trainFilename, testFilename = datasetFilenames('singles')

   with open(trainFilename, 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open(testFilename, 'r') as infile:
      testData = [processLine(line) for line in infile]

   return trainingData, testData


