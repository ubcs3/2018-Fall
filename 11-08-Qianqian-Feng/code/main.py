import argparse
import json
import os
import numpy as np

import utils
from naive_bayes import NaiveBayes

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(filename):
	return json.load(open(os.path.join('../data/', filename)))

def toString(l, mapper, f):
	l = l//f
	text = ''
	for i in range(len(l)):
		text += l[i]*(mapper[i]+' ')

	return text

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--do', required=True)

    io_args = parser.parse_args()
    action = io_args.do


    if action == 'load':
    	data = load_data('train.json')
    	utils.create_recipe_ingredients_matrix(data)

    if action == 'stats':
    	data = load_data('train.json')
    	X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisine_inverse_mapper = utils.load(data)
    	print("All loaded")
    	
    	count = np.bincount(y)
    	# for i in range(20):
    	# 	print(cuisine_inverse_mapper[i],":",count[i])
    	fig1 = plt.figure()
    	plt.bar(list(range(20)), count, align='edge', tick_label=list(cuisine_mapper.keys()))
    	plt.xticks(rotation=50)
    	plt.savefig('../figs/cuisine-count.jpg')
    	plt.close()

    	text_c = toString(count, cuisine_inverse_mapper, 200)
    	
    	wordcloud = WordCloud(max_words=20, background_color="white", collocations=False).generate(text_c)
    	plt.figure()
    	plt.imshow(wordcloud, interpolation="bilinear")
    	plt.axis("off")
    	plt.savefig('../figs/cuisine-cloud.jpg')
    	plt.close()

    	count_ingredients = np.array(np.sum(X, axis=0).T)
    	most_freq_ingre = np.argsort(count_ingredients, axis=None)
    	count_ingredients = np.sort(np.squeeze(count_ingredients),axis=None)

    	most_freq_ingre = most_freq_ingre[-100:]
    	plot_ingredients = count_ingredients[-100:]
    	most_freq_ingre = most_freq_ingre[::-1]
    	plot_ingredients = plot_ingredients[::-1]

    	x_label = list(map(lambda x: ingredient_inverse_mapper[x], most_freq_ingre))

    	fig2 = plt.figure(figsize=(20,10))
    	plt.bar(list(range(100)), plot_ingredients, align='edge', tick_label=x_label, color=list(map(lambda x: get_cmap(100)(x), list(range(100)))))
    	plt.xticks(rotation=50)
    	plt.savefig('../figs/ingredient-count.jpg')
    	plt.close()

    	text_i = ''.join([(plot_ingredients[i]//1000)*(x_label[i]+" ") for i in range(40)])
    	

    	cookingPotMask = np.squeeze(np.array(Image.open("../images/cooking-pot-clipart-black-and-white.png")))

    	wordcloud = WordCloud(background_color="white", collocations=False, max_words=100, mask=cookingPotMask)

    	wordcloud.generate(text_i)
    	plt.figure()
    	plt.imshow(wordcloud, interpolation="bilinear")
    	plt.axis("off")
    	plt.savefig('../figs/ingredient-cloud.jpg')
    	plt.close()

    if action == 'bayes':
    	data = load_data('train.json')
    	X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisine_inverse_mapper = utils.load(data)
    	
    	# cross-validation to pick the right beta for Laplase smoothing
    	print("Multinomial NaiveBayes")
    	print("======================")
    	for beta in range(0,10):
    		beta = beta/10.0
	    	error = 0
	    	for train_index, test_index in KFold(n_splits=5).split(X):
	    		# print('# of train_index:', len(train_index))
	    		# print('# of test_index:', len(test_index))
	    		X_train, X_test = X[train_index], X[test_index]
	    		y_train, y_test = y[train_index], y[test_index]
	    		model = BernoulliNB(alpha=beta)
	    		model.fit(X_train, y_train)

	    		y_pred = model.predict(X_test)
	    		v_error = np.mean(y_pred != y_test)

	    		error += v_error

	    	print("beta:", beta)
	    	print("error:", error/5)
	    	plt.scatter(beta,error/5)

    	plt.show()
    	plt.close()



    	# cross-validation to pick the right beta for Laplase smoothing
    	print("Complement NaiveBayes")
    	print("======================")
    	for beta in range(0,10):
    		beta = beta/10.0
	    	error = 0
	    	for train_index, test_index in KFold(n_splits=5).split(X):
	    		# print('# of train_index:', len(train_index))
	    		# print('# of test_index:', len(test_index))
	    		X_train, X_test = X[train_index], X[test_index]
	    		y_train, y_test = y[train_index], y[test_index]
	    		model = ComplementNB(alpha=beta)
	    		model.fit(X_train, y_train)

	    		y_pred = model.predict(X_test)
	    		v_error = np.mean(y_pred != y_test)

	    		error += v_error

	    	print("beta:", beta)
	    	print("error:", error/5)
	    	plt.scatter(beta,error/5)

    	plt.show()
    	plt.close()



    if action == "L1loss":
    	data = load_data('train.json')
    	X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisine_inverse_mapper = utils.load(data)
    	


    if action == "ensemble":
    	data = load_data('train.json')
    	X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisine_inverse_mapper = utils.load(data)

    	print("Bag of Bayes")
    	model = BaggingClassifier(BernoulliNB(alpha=0.1),max_samples=0.5,max_features=0.5)
    	error = 0
    	for train_index, test_index in KFold(n_splits=5).split(X):
    		# print('# of train_index:', len(train_index))
    		# print('# of test_index:', len(test_index))
    		X_train, X_test = X[train_index], X[test_index]
    		y_train, y_test = y[train_index], y[test_index]
    		model.fit(X_train, y_train)

    		y_pred = model.predict(X_test)
    		v_error = np.mean(y_pred != y_test)

    		error += v_error
    	print("error:", error/5)

    	print("========================")
    	print("Random Forest")
    	
    	for depth in range(30,31):
	    	model = RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=depth)
	    	error = 0
	    	for train_index, test_index in KFold(n_splits=5).split(X):
	    		# print('# of train_index:', len(train_index))
	    		# print('# of test_index:', len(test_index))
	    		X_train, X_test = X[train_index], X[test_index]
	    		y_train, y_test = y[train_index], y[test_index]
	    		model.fit(X_train, y_train)

	    		y_pred = model.predict(X_test)
	    		v_error = np.mean(y_pred != y_test)

	    		error += v_error
	    	print("depth:", depth)
	    	print("error:", error/5)
	    	plt.scatter(depth,error/5)

    	plt.show()






























    	
    	
