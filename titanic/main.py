# Ignore warnings
import warnings

from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import plot_utils
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

import load_data, datasets
# Configure visualisations
#%matplotlib inline

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8, 6


def main():
    # get titanic & test csv files as a DataFrame
    ds = load_data.DataSet()
    ds.load_files("train.csv","test.csv")
    full = ds.get_full()
    titanic = ds.get_titanic()

    '''
    #get information on data
    print('Datasets:', 'full:', full.shape, 'titanic:', titanic.shape)
    print(titanic.describe())
    plot_utils.plot_correlation_map(titanic)
    plot_utils.plots(titanic)
    '''

    #transform values into numerical types

    # Transform Sex into binary values 0 (female) and 1 (male)
    sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name='Sex')

    # Create a new variable for every unique value of Embarked (Embarked_C, Embarked_Q, Embarked_S)
    embarked = pd.get_dummies(full.Embarked, prefix='Embarked')

    # Create a new variable for every unique value of Embarked (Pclass_1, Pclass_2, Pclass_3)
    pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
    print(pclass.head())

    # Create dataset
    imputed = datasets.imputed(full)
    title = datasets.title(full)
    title.head()

    cabin= datasets.cabin(full)
    cabin.head()

    ticket = datasets.ticket(full)
    ticket.head()

    family = datasets.family(full)
    family.head()

    # Select which features/variables to include in the dataset from the list below:
    # imputed , embarked , pclass , sex , family , cabin , ticket
    full_X = pd.concat([imputed, embarked, cabin, sex], axis=1)
    full_X.head()

    train_valid_X = full_X[0:891]
    train_valid_y = titanic.Survived
    test_X = full_X[891:]
    train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.7)

    print(full_X.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape)

    plot_utils.plot_variable_importance(train_X, train_y)

    model = RandomForestClassifier(n_estimators=100)
   # model = SVC()
   # model = GradientBoostingClassifier()
   # model = KNeighborsClassifier(n_neighbors=3)
   # model = GaussianNB()
   # model = LogisticRegression()

    model.fit(train_X, train_y)

    print(model.score(train_X, train_y), model.score(valid_X, valid_y))
#    plot_utils.plot_model_var_imp(model, train_X, train_y)

    rfecv = RFECV(estimator=model, step=1, cv = StratifiedKFold().split(train_X, train_y), scoring='accuracy')
    rfecv.fit(train_X, train_y)

    test_Y = model.predict(test_X)
    passenger_id = full[891:].PassengerId

    test = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_Y})
    test['Survived'] = test['Survived'].astype(int)

   # test.head()
    test.to_csv('titanic_pred.csv', index=False)


if __name__ == "__main__":
    main()

'''
Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
Name - Name
Sex - Sex
Age - Age
Sibsp - Number of Siblings/Spouses Aboard
Parch - Number of Parents/Children Aboard
Ticket - Ticket Number
Fare - Passenger Fare
Cabin - Cabin
Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

'''