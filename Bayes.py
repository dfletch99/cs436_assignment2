import nltk
from nltk.stem import *	#word stemming
from nltk.corpus import stopwords
import os				#file io
import math 			#log
import re

def main():
	nltk.download('stopwords')

	[train_acc, test_acc] = classify(stop=False)
	[_, test_acc_stops] = classify(stop=True)

	fout = open('./results.txt', 'w', encoding='utf-8')
	#fout.write("Train:\tTest_with_stopwords\tTest_without_stopwords\n" + str(round(train_acc, 3)) + "\t" + str(round(test_acc, 3)) + "\t" + str(round(test_acc_stops, 3)))
	fout.write('{:<10s}{:>20s}{:>25s}\n'.format('Training', 'Test_with_stopwords', 'Test_without_stopwords'))
	fout.write('{:<11.3f}{:<22.3f}{:<25.3f}'.format(train_acc, test_acc, test_acc_stops))
	fout.close()
	print("ACCURACIES:\nTRAIN: " + str(train_acc) + "\nTEST_NOSTOPS: " + str(test_acc) + "\nTEST_STOPS: " + str(test_acc_stops))


def classify(stop):
	stemmer = SnowballStemmer('english')

	stops = set(stopwords.words('english'))

	#TRAINING DATA PROCESSING
	specialCharacters = ['subject:', '.', ',', '/', '\\', '!', '@', '`', '~', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}', ';', ':', '\'', '\"', '<', '>', '?', '-', '+', '=', '_']
	n_spam_train = 0
	n_ham_train = 0
	spam_bag_of_words = dict()
	ham_bag_of_words = dict()
	total_unique_words = 0
	total_spam_words = 0
	total_ham_words = 0

	#spam
	directory = "./train/spam/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			n_spam_train += 1
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					total_spam_words += 1
					if stem in spam_bag_of_words.keys():
						spam_bag_of_words[stem] += 1
					else:
						spam_bag_of_words[stem] = 1
						total_unique_words += 1
			fo.close()

	#ham
	directory = "./train/ham/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			n_ham_train += 1
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					total_ham_words += 1
					if stem in ham_bag_of_words.keys():
						ham_bag_of_words[stem] += 1
					else:
						ham_bag_of_words[stem] = 1
						if stem not in spam_bag_of_words.keys():
							total_unique_words += 1
			fo.close()


	#PRIORS
	prob_spam = n_spam_train/(n_spam_train + n_ham_train)
	prob_ham = n_ham_train/(n_spam_train + n_ham_train)

	#CONDITIONAL PROBABILITIES
	spam_probs = dict()
	ham_probs = dict()
	for key in spam_bag_of_words.keys():
		spam_probs[key] = (spam_bag_of_words[key] + 1) / (total_spam_words + total_unique_words)
		'''
		#include what would be 0-probability cases without laplace
		if key not in ham_bag_of_words.keys():
			ham_probs[key] = 1.0/(total_ham_words + total_unique_words)
			'''
	for key in ham_bag_of_words.keys():
		ham_probs[key] = (ham_bag_of_words[key] + 1) / (total_spam_words + total_unique_words)
		#0-prob
		'''
		if key not in spam_bag_of_words.keys():
			spam_probs[key] = 1.0/(total_spam_words + total_unique_words)
		'''

	#TEST DATA
	n_spam_test = 0
	n_spam_correct = 0
	n_ham_test = 0
	n_ham_correct = 0

	directory = "./train/spam/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			bag_of_words = []
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					# if stem in spam_bag_of_words.keys() or stem in ham_bag_of_words.keys():
					#	bag_of_words.append(stem)
					bag_of_words.append(stem)
			fo.close()

			# classifying probabilities
			classify_spam = math.log(prob_spam)
			classify_ham = math.log(prob_ham)
			for word in bag_of_words:
				if word in spam_bag_of_words.keys():
					classify_spam += math.log(spam_probs[word])
				else:
					classify_spam += math.log(1.0 / (total_spam_words + total_unique_words))
				if word in ham_bag_of_words.keys():
					classify_ham += math.log(ham_probs[word])
				else:
					classify_ham += math.log(1.0 / (total_ham_words + total_unique_words))
			if classify_spam > classify_ham:
				n_spam_correct += 1

	directory = "./train/ham/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			bag_of_words = []
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					# if stem in spam_bag_of_words.keys() or stem in ham_bag_of_words.keys():
					#	bag_of_words.append(stem)
					bag_of_words.append(stem)
			fo.close()

			# classifying probabilities
			classify_spam = math.log(prob_spam)
			classify_ham = math.log(prob_ham)
			for word in bag_of_words:
				if word in spam_bag_of_words.keys():
					classify_spam += math.log(spam_probs[word])
				else:
					classify_spam += math.log(1.0 / (total_spam_words + total_unique_words))
				if word in ham_bag_of_words.keys():
					classify_ham += math.log(ham_probs[word])
				else:
					classify_ham += math.log(1.0 / (total_ham_words + total_unique_words))
			if classify_spam < classify_ham:
				n_ham_correct += 1

	train_acc = (n_spam_correct + n_ham_correct) / (n_spam_train + n_ham_train)
	n_spam_correct = 0
	n_ham_correct = 0

	directory = "./test/spam/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			n_spam_test += 1
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			bag_of_words = []
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					#if stem in spam_bag_of_words.keys() or stem in ham_bag_of_words.keys():
					#	bag_of_words.append(stem)
					bag_of_words.append(stem)
			fo.close()

			#classifying probabilities
			classify_spam = math.log(prob_spam)
			classify_ham = math.log(prob_ham)
			for word in bag_of_words:
				if word in spam_bag_of_words.keys():
					classify_spam += math.log(spam_probs[word])
				else:
					classify_spam += math.log(1.0 / (total_spam_words + total_unique_words))
				if word in ham_bag_of_words.keys():
					classify_ham += math.log(ham_probs[word])
				else:
					classify_ham += math.log(1.0 / (total_ham_words + total_unique_words))
			if classify_spam > classify_ham:
				n_spam_correct += 1

	directory = "./test/ham/"
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			n_ham_test += 1
			fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
			bag_of_words = []
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					#if stem in spam_bag_of_words.keys() or stem in ham_bag_of_words.keys():
					#	bag_of_words.append(stem)
					bag_of_words.append(stem)
			fo.close()

			# classifying probabilities
			classify_spam = math.log(prob_spam)
			classify_ham = math.log(prob_ham)
			for word in bag_of_words:
				if word in spam_bag_of_words.keys():
					classify_spam += math.log(spam_probs[word])
				else:
					classify_spam += math.log(1.0 / (total_spam_words + total_unique_words))
				if word in ham_bag_of_words.keys():
					classify_ham += math.log(ham_probs[word])
				else:
					classify_ham += math.log(1.0 / (total_ham_words + total_unique_words))
			if classify_spam < classify_ham:
				n_ham_correct += 1

	test_acc = (n_spam_correct + n_ham_correct) / (n_spam_test + n_ham_test)
	n_spam_correct = 0
	n_ham_correct = 0
	return [train_acc, test_acc]


if __name__ == '__main__':
	main()