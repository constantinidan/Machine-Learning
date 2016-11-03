## Python version 3.5.0 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import sklearn

#####
##### Partie 1 : Statistiques descriptives
#####


### For generating a coloured matrix correlation
def display(corr):
	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

	# Title
	ax.set_title("Correlation matrix on continuous variables")

	# Draw the heatmap with the mask and correct aspect ratio
	seaborn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
				square=True, xticklabels=5, yticklabels=5,
				linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
	# Displaying
	seaborn.heatmap(corr)


### Classes des velos
class bikes():
	def __init__(self, path):
		""" Initialization """
		self.df = pd.DataFrame.from_csv(path)
		self.preprocess()

	def preprocess(self):
		""" New variables """
		self.df['hour'] = self.df.index.map(lambda x: x.hour)
		self.df['dayofweek'] = self.df.index.map(lambda x: x.dayofweek)
		self.df['month'] = self.df.index.map(lambda x: x.month)
		self.df['year'] = self.df.index.map(lambda x: x.year)
		self.df['floorTemp'] = self.df['temp'].map(lambda x : int(x))


	def corrMatrix(self):
		""" Correlation matrix on continuous variables """
		continuousVariable = ["temp","atemp", "humidity", "windspeed", 
								"casual","registered", "count"]
		display(self.df[continuousVariable].corr())
		plt.show()

	def histograms(self):
		""" Histogramms """
		self.df.hist()
		plt.show()

	def showBoxPlot(self, variable, absciss):
		""" Boxplots of 'variable' grouped by 'absciss' """
		self.df.boxplot(column=variable, by=absciss, showmeans=True)
		plt.show()

	def meanStd(self,variable, absciss):
		""" Mean and Std of 'variable' grouped by 'absciss' """
		grouped = self.df.groupby(absciss)
		return grouped[variable].agg({'mean': np.mean, 'interval': lambda x : 1.96*np.std(x,ddof=1)/np.sqrt(len(x))})

	def temp(self):
		""" Displays the mean and standard deviation for each temperature. """
		tmpdf = self.meanStd('count', 'floorTemp')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne du nombre total de locations en fonction de la temperature')
		plt.xlabel('temperature')
		plt.show()

	def season(self):
		""" Displays the mean and standard deviation for each season. """
		tmpdf = self.meanStd('count', 'season')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks( range(4), ('Printemps', 'Ete', 'Automne', 'Hiver'), rotation=0)
		plt.title('Moyenne du nombre total de locations en fonction de la saison')
		plt.xlabel('saison')
		plt.show()

	def month(self):
		""" Displays the mean and standard deviation for each month. """
		tmpdf = self.meanStd('count', 'month')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne du nombre total de locations en fonction du mois')
		plt.xlabel('mois')
		plt.show()

	def dayofweek(self):
		""" Displays the mean and standard deviation for each day of the week. """
		tmpdf = self.meanStd('registered', 'dayofweek')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(range(7), ('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 
					'Samedi', 'Dimanche'), rotation=0)
		plt.title('Moyenne des locations par des abonnes en fonction du jour de la semaine')
		plt.xlabel('jour de la semaine')
		plt.show()

		tmpdf = self.meanStd('casual', 'dayofweek')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(range(7), ('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 
					'Samedi', 'Dimanche'), rotation=0)
		plt.title('Moyenne des locations par des non-abonnes en fonction du jour de la semaine')
		plt.xlabel('jour de la semaine')
		plt.show()

	def hour(self):
		""" Displays the mean and standard deviation for each hour of the day. """
		tmpdf = self.meanStd('registered','hour')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne des locations par des abonnes en fonction de l\'heure')
		plt.xlabel('heure de la journee')
		plt.show()

		tmpdf = self.meanStd('casual', 'hour')
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne des locations par des non-abonnes en fonction de l\'heure')
		plt.xlabel('heure de la journee')
		plt.show()

	def workingday(self):
		self.showBoxPlot(['casual', 'registered'], 'workingday')

	def holiday(self):
		self.showBoxPlot(['casual', 'registered'], 'holiday')

	def weather(self):
		""" Analysing the weather. """
		tmpdf = self.meanStd('registered','weather')
		tmpdf.drop(4,inplace=True)
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne des locations faites par des abonnes en fonction de la meteo')
		plt.xlabel('meteo')
		plt.show()

		tmpdf = self.meanStd('casual','weather')
		tmpdf.drop(4,inplace=True)
		tmpdf['mean'].plot(yerr= tmpdf['interval'], kind='bar')
		plt.xticks(rotation=0)
		plt.title('Moyenne des locations faites par des abonnes en fonction de la meteo')
		plt.xlabel('meteo')
		plt.show()


data = bikes('data.csv')
data.corrMatrix()
data.temp()
data.season()
data.month()
data.dayofweek()
data.hour()
data.workingday()
data.holiday()
data.weather()



#####
##### Partie 2 : Machine Learning
#####


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve,cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble, linear_model


##### Training and testing data
inputVariables = ['season', 'holiday', 'workingday', 'weather', 'temp', 
					'atemp', 'humidity', 'windspeed','dayofweek', 'hour',
					'month', 'year'] 

## Splitting the data.
X_train_sum, X_test_sum, y_train_sum, y_test_sum = train_test_split(data.df[inputVariables].values, data.df['count'].values, test_size=0.3, random_state=0)



##### Linear Regression 
#####
regr = linear_model.LinearRegression()
scores = cross_val_score(regr, data.df[inputVariables].values, data.df['count'].values)
print("Linear Regression cross validation score: ", scores.mean())
regr.fit(X_train_sum, y_train_sum)
print("Linear Regression training score: ", regr.score(X_train_sum, y_train_sum))
print("Linear Regression testing score: ", regr.score(X_test_sum, y_test_sum))



##### Kernel Ridge and Support Vector Regression
#####
## Finding the best parameters
alpha=[1,1e-1,1e-2,1e-3]
for a in alpha:
	kr = KernelRidge(kernel='rbf', alpha=a)
	kr.fit(X_train_sum, y_train_sum)
	print("Kernel Ridge train score: ", kr.score(X_train_sum, y_train_sum), " for alpha = %s" %a)
	print("Kernel Ridge test score: ", kr.score(X_test_sum, y_test_sum), " for alpha = %s" %a)


### Using GridSearchCV
param_grid = { 
	'alpha': [1, 1e-1, 1e-2]
	"gamma": np.logspace(-2, 2, 5)
}
GSKernelRidge = GridSearchCV(KernelRidge(kernel='rbf'), param_grid=param_grid)
GSKernelRidge.fit(X_train_sum, y_train_sum)





kr = KernelRidge(kernel='rbf', alpha=1e-3)
scores = cross_val_score(kr, data.df[inputVariables].values, data.df['count'].values, cv =3)
print("Kernel Ridge cross validation score: ", scores.mean())

## Finding the best parameters
C=[10,100,1000]
for c in C:
	svr = SVR(kernel='rbf', C=c, epsilon=0.1)
	svr.fit(X_train_sum, y_train_sum)
	print("SVR train score: ", svr.score(X_train_sum, y_train_sum), " for C = %s" %c)
	print("SVR test score: ", svr.score(X_test_sum, y_test_sum), " for C = %s" %c)

svr = SVR(kernel='rbf', C=1000, epsilon=0.1)
scores = cross_val_score(svr, data.df[inputVariables].values, data.df['count'].values, cv=4)
print("SVR cross validation score: ", scores.mean())



##### Gradient Boosting Regression
#####
params = {'n_estimators': 2000, "learning_rate":0.1}
GBR = ensemble.GradientBoostingRegressor(**params)
GBR.fit(X_train_sum, y_train_sum)
print("Gradient Boosting training score: ", GBR.score(X_train_sum, y_train_sum))
print("Gradient Boosting testing score: ", GBR.score(X_test_sum, y_test_sum))
scores = cross_val_score(GBR, data.df[inputVariables].values, data.df['count'].values, cv=5)
print("Gradient Boosting cross validation score: ", scores.mean())



##### Random Forest Regressor
#####
params = {'n_estimators': 40}
RFR = ensemble.RandomForestRegressor(**params)
RFR.fit(X_train_sum, y_train_sum)
print("Random Forest training score: ", RFR.score(X_train_sum, y_train_sum))
print("Random Forest testing score: ", RFR.score(X_test_sum, y_test_sum))
scores = cross_val_score(RFR, data.df[inputVariables].values, data.df['count'].values, cv=5)
print("Random Forest cross validation score: ", scores.mean())

### Curve of score in fonction of the nomber of estimators
scoresCount = []
estimators = [1,3,5,7,10,30,50,80, 100]
for e in estimators:
	params = {'n_estimators': e}
	RFR = ensemble.RandomForestRegressor(**params)
	RFR.fit(X_train_sum, y_train_sum)
	scoresCount.append(RFR.score(X_test_sum, y_test_sum))

plt.plot(estimators, scoresCount)
plt.xlabel('Nombre d\'estimateurs')
plt.ylabel('R2_score')
plt.show()


##### Model improvement
## Splitting into two models: registered and casual to see if there is improvement
X_train, X_test, y_train, y_test = train_test_split(data.df[inputVariables].values, data.df[['registered', 'casual']].values, test_size=0.3, random_state=0)
y_train_reg = y_train[:,0]
y_train_cas = y_train[:,1]
y_test_reg = y_test[:,0]
y_test_cas = y_test[:,1]

params = {'n_estimators': 50}
RFR_reg = ensemble.RandomForestRegressor(**params)
RFR_reg.fit(X_train, y_train_reg)
RFR_cas = ensemble.RandomForestRegressor(**params)
RFR_cas.fit(X_train, y_train_cas)
print('2-M Random Forest test score: ',r2_score(RFR_cas.predict(X_test) + RFR_reg.predict(X_test), y_test_reg+y_test_cas))








