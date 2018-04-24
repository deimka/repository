import pandas as pd

def imputed(full):
    imputed = pd.DataFrame()

    # Fill missing values of Age with the average of Age (mean)
    imputed['Age'] = full.Age.fillna(full.Age.mean())

    # Fill missing values of Fare with the average of Fare (mean)
    imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
    return imputed



def title(full):
    title = pd.DataFrame()
    # we extract the title from each name
    title['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    title['Title'] = title.Title.map(Title_Dictionary)
    title = pd.get_dummies(title.Title)
    # title = pd.concat( [ title , titles_dummies ] , axis = 1 )
    return title


def cabin(full):
    cabin = pd.DataFrame()

    # replacing missing cabins with U (for Uknown)
    cabin['Cabin'] = full.Cabin.fillna('U')

    # mapping each Cabin value with the cabin letter
    cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin')
    return cabin


def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else:
        return 'XXX'


def ticket(full):
    ticket = pd.DataFrame()

    # Extracting dummy variables from tickets:
    ticket['Ticket'] = full['Ticket'].map(cleanTicket)
    ticket = pd.get_dummies(ticket['Ticket'], prefix='Ticket')

    return ticket


def family(full):
    family = pd.DataFrame()

    # introducing a new feature : the size of families (including the passenger)
    family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

    # introducing other features based on the family size
    family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    family['Family_Small'] = family['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    family['Family_Large'] = family['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return family

