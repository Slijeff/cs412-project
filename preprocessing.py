from typing import Tuple
from pandas import DataFrame
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def preprocess_raw(df : DataFrame):
    """This function takes the raw input of pd.read_csv() on the original data and do all the preprocessing on it
    Current preprocessing: categorical to numeric, PCA
    Args:
        df (DataFrame): the raw input

    Returns:
        X_train, X_test, y_train, y_test
    """
     # transform categorical columns to pandas categorical data type
    for cname in df.select_dtypes("object").columns:
        df[cname] = df[cname].astype("category")
    nominal_names = ['State', 'Sex', 'PhysicalActivities','HadAngina','HadStroke','HadAsthma','HadSkinCancer', 'HadCOPD'
                 ,'HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadDiabetes','DeafOrHardOfHearing','BlindOrVisionDifficulty'
                 ,'DifficultyConcentrating','DifficultyWalking','DifficultyDressingBathing','DifficultyErrands','ChestScan','RaceEthnicityCategory'
                 ,'AlcoholDrinkers', 'HIVTesting','FluVaxLast12','PneumoVaxEver','TetanusLast10Tdap','HighRiskLastYear','CovidPos']
    ordinal_names = ['GeneralHealth', 'LastCheckupTime', 'RemovedTeeth','SmokerStatus','ECigaretteUsage','AgeCategory']
    assert len(nominal_names) + len(ordinal_names) + 1 == len(df.select_dtypes("category").columns)
    noms = pd.get_dummies(df[nominal_names], dtype=pd.UInt8Dtype())
    y = pd.get_dummies(df[["HadHeartAttack"]], dtype=pd.UInt8Dtype(), drop_first=True)
    df = df.drop(nominal_names + ["HadHeartAttack"], axis=1).join(noms)
    general_health_mapper = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1, 'Very good': 0}
    lastcheckup_mapper = {'Within past year (anytime less than 12 months ago)': 3, 'Within past 2 years (1 year but less than 2 years ago)': 2
                        ,'Within past 5 years (2 years but less than 5 years ago)': 1, '5 or more years ago': 0}
    removed_teeth_mapper = {'None of them': 3, '1 to 5': 2, '6 or more, but not all': 1, 'All': 0}
    somker_status_mapper = {'Never smoked': 3, 'Former smoker': 2, 'Current smoker - now smokes some days': 1
                            , 'Current smoker - now smokes every day': 0}
    ecigar_mapper = {'Never used e-cigarettes in my entire life': 3, 'Not at all (right now)': 2, 'Use them some days': 1,
                    'Use them every day': 0}
    age_mapper = {
        "Age 18 to 24": 12,
        "Age 25 to 29": 11,
        "Age 30 to 34": 10,
        "Age 35 to 39": 9,
        "Age 40 to 44": 8,
        "Age 45 to 49": 7,
        "Age 50 to 54": 6,
        "Age 55 to 59": 5,
        "Age 60 to 64": 4,
        "Age 65 to 69": 3,
        "Age 70 to 74": 2,
        "Age 75 to 79": 1,
        "Age 80 or older": 0
    }
    mappers = [general_health_mapper, lastcheckup_mapper, removed_teeth_mapper, somker_status_mapper, ecigar_mapper, age_mapper]
    for i, cname in enumerate(ordinal_names):
        df[cname] = df[cname].replace(mappers[i])
    
    # if a feature column is the same 98% of the time, we'll remove it
    thresh = 0.98
    sel = VarianceThreshold(threshold=(thresh * (1 - thresh)))
    sel.set_output(transform='pandas')
    sel.fit(df)
    df = sel.transform(df)

    # I was going to do PCA here, but since PCA changes the value of the data, I would do it before model fitting

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.8, random_state=0)
    return  X_train, X_test, y_train, y_test