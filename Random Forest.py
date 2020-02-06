#Importação de pandas
import pandas as pd

#Leitura dos data bases
train = pd.read_csv('train.csv')
previsores = train.iloc[:, 0:8].values
classe = train.iloc[:, 8].values
test = pd.read_csv('test.csv')

#Importação do SimpleImputer
from sklearn.impute import SimpleImputer
#Definição de variável para imputer
imputer_age = SimpleImputer()
#Aplicação do imputer na coluna Age do df previsores
imputer_age = imputer_age.fit(previsores[:, 3:4])
#Atualização do df previsores com imputer aplicado
previsores[:, 3:4] = imputer_age.transform(previsores[:, 3:4])

#Imputer para df test para coluna Age
imputer_age_test = SimpleImputer()
imputer_age_test = imputer_age_test.fit(test.iloc[:, 3:4])
test.iloc[:, 3:4] = imputer_age_test.transform(test.iloc[:, 3:4])

#Imputer para df previsores para coluna Fare
imputer_fare = SimpleImputer()
imputer_fare = imputer_fare.fit(previsores[:, 6:7])
previsores[:, 6:7] = imputer_fare.transform(previsores[:, 6:7])

#Imputer para df test para coluna Fare
imputer_fare_test = SimpleImputer()
imputer_fare_test = imputer_fare_test.fit(test.iloc[:, 6:7])
test.iloc[:, 6:7] = imputer_fare_test.transform(test.iloc[:, 6:7])

#Imputer para df provisores para coluna Embarked
imputer_embarked = SimpleImputer(strategy="most_frequent")
imputer_embarked = imputer_embarked.fit(previsores[:, 7:8])
previsores[:, 7:8] = imputer_embarked.transform(previsores[:, 7:8])

#Imputer para df test para coluna Embarked
imputer_embarked_test = SimpleImputer(strategy="most_frequent")
imputer_embarked_test = imputer_embarked_test.fit(test.iloc[:, 7:8])
test.iloc[:, 7:8] = imputer_embarked_test.transform(test.iloc[:, 7:8])

#Importações de OneHotEncoder e ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#Definição de variável do ColumnTransformer com parâmetros já setados
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2, 7])], remainder='passthrough')
#Aplicação no df previsores
previsores = columnTransformer.fit_transform(previsores)
#Aplicação no df test
test = columnTransformer.fit_transform(test)

#Importação do Random Forest
from sklearn.ensemble import RandomForestClassifier
#Definição de variável para Random Forest
algoritimo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
#Aplicação do Random Forest nos df's previcores e classe)
algoritimo.fit(previsores, classe)
#Criação de variável com resultados do Random Forest
previsoes = algoritimo.predict(test)

#Converte numpy array em pandas df
resultado = pd.DataFrame(previsoes)
#Exporta df para csv file
resultado.to_csv('resultados Random Forest.csv', index=False, header=False)
