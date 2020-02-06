#Importações
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

#Leitura dos data bases
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#Criação de transformador numérico e categórico
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Criação de transformador personalizado para coluna "Name" 
#(substitui o título pelo vetor correspondente)
class TitleSelector(BaseEstimator, TransformerMixin):
    def __init__( self):
        self.dict_title = {
            "Capt":       0,
            "Col":        0,
            "Major":      0,
            "Jonkheer":   1,
            "Don":        1,
            "Sir" :       1,
            "Dr":         0,
            "Rev":        0,
            "the Countess":1,
            "Dona":       1,
            "Mme":        2,
            "Mlle":       3,
            "Ms":         2,
            "Mr" :        4,
            "Mrs" :       2,
            "Miss" :      3,
            "Master" :    5,
            "Lady" :      1
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for i, name in enumerate(X["Name"]):
            for title in self.dict_title.keys():
                if title in name:
                    X["Name"][i] = self.dict_title[title]
                    break
                
            assert X["Name"][i] in self.dict_title.values()
            
        return X

name_transformer = Pipeline(steps=[
    ('name', TitleSelector()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Colunas numéricas
num_cols = ["Age", "Fare", ]
#Colunas categóricas
cat_cols = ["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"]
#Colunas totais
cols = num_cols + cat_cols + ["Name"]

#Criação de ColumnTransformer e fit e transform
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_cols),
    ('name', name_transformer, ["Name"]),
    ('cat', categorical_transformer, cat_cols),
])

X_train = preprocessor.fit_transform(df_train[cols])
y_train = df_train["Survived"].values

#Criação de rede neural com 2 camadas ocultas 
model = Sequential()
model.add(Dense(32, input_dim=858, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))    

#Treinamento da rede neural
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=8)

X_test = preprocessor.transform(df_test[cols])
y_pred = model.predict_classes(X_test)

#Criação de arquivo para predições
df_pred = pd.DataFrame(df_test["PassengerId"])
df_pred["Survived"] = y_pred
df_pred.to_csv("submission.csv", index=False)
    
