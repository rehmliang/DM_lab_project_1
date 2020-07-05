from preprocess_pipeline.datautils import *

name = "adult"
protectedIndex = 1 # gender
protectedValue = 0 # female

employers = ('Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
             'Local-gov', 'State-gov', 'Without-pay', 'Never-worked')
maritals = ('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse')
occupations = ('Tech-support', 'Craft-repair', 'Other-service', 'Sales',
               'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
               'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
               'Protective-serv', 'Armed-Forces')
races = ('White', 'Asian-Pac-Islander','Amer-Indian-Eskimo', 'Other', 'Black')
sexes = ('Female', 'Male')
countries = ('United-States', 'Cambodia', 'England', 'Puerto-Rico',
            'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
            'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
            'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
            'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
            'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
            'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands')


featureNames = (('age','sex') + employers + ('education',) + maritals + occupations +
               races + ('capital_gain','capital_loss','hr_per_week') + countries)


def processLine(line):
   values = line.strip().split(', ')
   (age, employer, _, _, education, marital, occupation, _, race, sex,
      capital_gain, capital_loss, hr_per_week, country, income) = values

   # put sex second so I know what index it is
   point = ([int(age), 0 if sex=='Female' else 1] + vectorize(employer, employers) + [int(education)] +
             vectorize(marital, maritals) + vectorize(occupation, occupations) + vectorize(race, races) +
            [int(capital_gain), int(capital_loss), int(hr_per_week)] + vectorize(country, countries))
   label = 1 if income[0] == '>' else -1

   return tuple(point), label


def load():
   trainFilename, testFilename = datasetFilenames('adult')

   with open(trainFilename, 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open(testFilename, 'r') as infile:
      testData = [processLine(line) for line in infile]

   return trainingData, testData


