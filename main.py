import pickle as pkl
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import functions as func
import net as mynet
import hyperparameters as hparams

#HYPERPARAMETERS
numEpochs = hparams.numEpochs
batchSize = hparams.batchSize
learningRate = hparams.learningRate

#DATA LOADING
df = pkl.load(open(hparams.datasetFile, "rb"))
df.dropna(inplace = True) #remove rows with nans
df = df.sample(frac = 1)
trainingData = torch.tensor(df.values) #trainingData used as temporary holder
labels = torch.tensor(df["t_gas"].values)
del df["t_gas"]
features = df.columns

#DATA SETUP
testData = torch.tensor(df.iloc[1000000:1080000,:].values)
testLabels = labels[1000000:1080000]
valData = torch.tensor(df.iloc[:200000,:].values)
valLabels = labels[:200000]
trainingData = torch.tensor(df.iloc[200000:200000 + int(hparams.proportionTrainingDatasetUsed * 800000),:].values)
trainingLabels = labels[200000:200000 + int(hparams.proportionTrainingDatasetUsed * 800000)]

#DATA PREPROCESSING: Transform training, validation data
twoDScaler = StandardScaler(copy = False)
twoDScaler.fit_transform(trainingData)
trainingLabelsMean, trainingLabelsStdev = torch.mean(trainingLabels), torch.std(trainingLabels)
trainingLabels = (trainingLabels - trainingLabelsMean) / trainingLabelsStdev
twoDScaler.transform(valData)
valLabels = (valLabels - trainingLabelsMean) / trainingLabelsStdev

with open(hparams.outputFile, "w") as ext, open(hparams.outputFile2, "w") as ext2, open(hparams.outputFile3, "w") as ext3, open(hparams.outputFile4, "w") as ext4, open(hparams.outputFile5, "w") as ext5:
	#TRAINING/TESTING CYCLE
	net = mynet.Net()
	net = func.trainModel(net, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4)
	
	#CROSSVALIDATION SETUP
	folds = []
	firstTestingLoss = func.testModel(net, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5)
	bestTestingLoss = firstTestingLoss
	bestNet = net
	folds.append(firstTestingLoss)
	if hparams.modelTesting and not hparams.crossVal:
		print(firstTestingLoss)

	#CROSS VALIDATION
	if hparams.crossVal:
		#fold 2 of 5
		#DATA SETUP
		testData = torch.tensor(df.iloc[1000000:1080000,:].values)
		testLabels = labels[1000000:1080000]
		valData = torch.tensor(df.iloc[200000:400000,:].values) 
		valLabels = labels[200000:400000]
		trainingData = torch.cat((torch.tensor(df.iloc[0:200000,:].values), torch.tensor(df.iloc[400000:,:].values)))
		trainingLabels = torch.cat((labels[:200000], labels[400000:]))
		#DATA PREPROCESSING: Transform training, validation data
		twoDScaler = StandardScaler(copy = False)
		twoDScaler.fit_transform(trainingData)
		trainingLabelsMean, trainingLabelsStdev = torch.mean(trainingLabels), torch.std(trainingLabels)
		trainingLabels = (trainingLabels - trainingLabelsMean) / trainingLabelsStdev
		twoDScaler.transform(valData)
		valLabels = (valLabels - trainingLabelsMean) / trainingLabelsStdev
		#TRAINING/TESTING CYCLE
		net2 = mynet.Net()
		net2 = func.trainModel(net2, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4)
		currentTestingLoss = func.testModel(net2, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5)
		#crossvalidation stuff
		folds.append(currentTestingLoss)
		if (currentTestingLoss < bestTestingLoss):
			bestNet = net2
			bestTestingLoss = currentTestingLoss

		#fold 3 of 5
		#DATA SETUP
		testData = torch.tensor(df.iloc[1000000:1080000,:].values)
		testLabels = labels[1000000:1080000]
		valData = torch.tensor(df.iloc[420000:630000,:].values)
		valLabels = labels[420000:630000]
		trainingData = torch.cat((torch.tensor(df.iloc[0:420000,:].values), torch.tensor(df.iloc[630000:,:].values)))
		trainingLabels = torch.cat((labels[:420000], labels[630000:]))
		#DATA PREPROCESSING
		twoDScaler = StandardScaler(copy = False)
		twoDScaler.fit_transform(trainingData)
		trainingLabelsMean, trainingLabelsStdev = torch.mean(trainingLabels), torch.std(trainingLabels)
		trainingLabels = (trainingLabels - trainingLabelsMean) / trainingLabelsStdev
		twoDScaler.transform(valData)
		valLabels = (valLabels - trainingLabelsMean) / trainingLabelsStdev
		#TRAINING/TESTING CYCLE
		net3 = mynet.Net()
		net3 = func.trainModel(net3, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4)
		currentTestingLoss = func.testModel(net3, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5)
		#crossvalidation stuff
		if (currentTestingLoss < bestTestingLoss):
			bestNet = net3
			bestTestingLoss = currentTestingLoss
		folds.append(currentTestingLoss)	

		#fold 4 of 5
		#DATA SETUP
		testData = torch.tensor(df.iloc[1000000:1080000,:].values)
		testLabels = labels[1000000:1080000]
		valData = torch.tensor(df.iloc[630000:840000,:].values) 
		valLabels = labels[630000:840000]
		trainingData = torch.cat((torch.tensor(df.iloc[0:630000,:].values), torch.tensor(df.iloc[840000:,:].values)))
		trainingLabels = torch.cat((labels[:630000], labels[840000:]))
		#DATA PREPROCESSING
		twoDScaler = StandardScaler(copy = False)
		twoDScaler.fit_transform(trainingData)
		trainingLabelsMean, trainingLabelsStdev = torch.mean(trainingLabels), torch.std(trainingLabels)
		trainingLabels = (trainingLabels - trainingLabelsMean) / trainingLabelsStdev
		twoDScaler.transform(valData)
		valLabels = (valLabels - trainingLabelsMean) / trainingLabelsStdev
		#TRAINING/TESTING CYCLE
		net4 = mynet.Net()
		net4 = func.trainModel(net4, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4)
		currentTestingLoss = func.testModel(net4, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5)
		#crossvalidation stuff
		if (currentTestingLoss < bestTestingLoss):
			bestNet = net4
			bestTestingLoss = currentTestingLoss
		folds.append(currentTestingLoss)

		#fold 5 of 5
		#DATA SETUP
		testData = torch.tensor(df.iloc[1000000:1080000,:].values)
		testLabels = labels[1000000:1080000]
		valData = torch.tensor(df.iloc[840000:,:].values) 
		valLabels = labels[840000:]
		trainingData = torch.tensor(df.iloc[840000:,:].values)
		trainingLabels = labels[840000:]
		#DATA PREPROCESSING
		twoDScaler = StandardScaler(copy = False)
		twoDScaler.fit_transform(trainingData)
		trainingLabelsMean, trainingLabelsStdev = torch.mean(trainingLabels), torch.std(trainingLabels)
		trainingLabels = (trainingLabels - trainingLabelsMean) / trainingLabelsStdev
		twoDScaler.transform(valData)
		valLabels = (valLabels - trainingLabelsMean) / trainingLabelsStdev
		#TRAINING/TESTING CYCLE
		net5 = mynet.Net()
		net5 = func.trainModel(net5, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4)
		currentTestingLoss = func.testModel(net5, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5)
		#crossvalidation stuff
		if (currentTestingLoss < bestTestingLoss):
			bestNet = net5
			bestTestingLoss = currentTestingLoss
		folds.append(currentTestingLoss)

	#OUTPUT
	if hparams.crossVal:
			print("Test loss of each fold: ", folds, file = ext)
	

ext.close()
ext2.close()
ext3.close()
ext4.close()
ext5.close()

#SAVING MODEL
torch.save(bestNet, hparams.modelFile)

###################################################



