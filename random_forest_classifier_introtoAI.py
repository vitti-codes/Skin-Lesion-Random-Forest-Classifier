import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('augmented_dataframe.csv')

sex_dict = {'male': 0, 'female': 1, 'unknown': 2}
df['sex_2'] = df['sex'].apply(sex_dict.get).astype(float)

dx_type_dict = {'histo': 0, 'consensus': 1, 'follow_up': 2, 'confocal': 3}
df['dx_type2'] = df['dx_type'].apply(dx_type_dict.get).astype(float)

localization_dict = {'scalp': 0, 'ear': 1, 'face': 2, 'back': 3, 'trunk': 4, 'chest': 5, 'upper extremity': 6,
'abdomen': 7, 'unknown': 8, 'lower extremity': 9, 'genital': 10, 'neck': 11, 'hand': 12, 'foot': 13, 'acral': 14}
df['loc_2'] = df['localization'].apply(localization_dict.get).astype(float)

X = df[['age', 'sex_2', 'loc_2', 'dx_type2']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, shuffle=True)

#print("training set: ", X_train.shape, y_train.shape)
#print("test set: ", X_test.shape, y_test.shape)


#CREATING THE RANDOM FOREST
rfc = RandomForestClassifier(criterion= 'entropy', random_state=24, n_estimators=75)
rfc.fit(X_train, y_train)

#evaluating the training set
y_pred = rfc.predict(X_test)

print("Training Set f1-Score: ", f1_score(y_test, y_pred, average='micro'))
print("Training set accuracy: ", accuracy_score(y_test, y_pred))


#finding the best parameters for the random forest: 75
param_grid = {'n_estimators': [10, 25, 50, 75, 100]}
gs = GridSearchCV(rfc, param_grid, cv = 5)
gs.fit(X_train, y_train)
print("best params: ", gs.best_params_)

#converting dataframe to numpy array

#plotting the data in a scater plot
#plt.scatter(df['colm. 1'], df['column 2', c=df['class column']])
#respectively x and y axis with color of the class
#plt.xlabel('name of x axis eg. AGE')#labeling x-axis
#plt.ylabel('name of y axis eg. GENDER')
