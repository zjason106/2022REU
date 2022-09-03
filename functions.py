import torch
import torch.optim as optim
import net as mynet
import torch.nn as nn
import torchmetrics as tm
import hyperparameters as hparams

#HYPERPARAMETERS
numEpochs = hparams.numEpochs
batchSize = hparams.batchSize
learningRate = hparams.learningRate
crossVal = hparams.crossVal

def trainModel(net, trainingData, trainingLabels, valData, valLabels, ext, ext2, ext3, ext4):
	optimizer = optim.Adam(net.parameters(), lr = learningRate)

	#used for early stopping
	currentBestValidationLoss = 0
	bestNet = mynet.Net()

	for epoch in range(numEpochs):
		i = 0
		while i * batchSize < trainingData.shape[0]: #for each batch
			#SETTING UP DATA
			data = trainingData[i*batchSize:i*batchSize + batchSize] #data is the current batch
			target = trainingLabels[i*batchSize:i*batchSize + batchSize]

			#FORWARD
			out = net(data)
			computed = torch.select(out,1,0)
			mse = nn.MSELoss()
			loss = mse(computed, target)

			#BACKWARD
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			i += 1

		if hparams.epochTrainingLoss:
			#COMPUTE METRICS
			trainOut = net(trainingData)
			computed = torch.select(trainOut,1,0) #changes shape from (valData.size, 1) to (valData.size)

			#REPORT METRICS
			mse = nn.MSELoss()
			loss = mse(computed, trainingLabels)
			print(epoch, loss.item(), file = ext, sep = ",")
		
		if hparams.epochValLoss:
			#COMPUTE METRICS
			valOut = net(valData)
			computed = torch.select(valOut, 1, 0) #changes shape from (valData.size, 1) to (valData.size)

			#REPORT METRICS
			mse = nn.MSELoss()
			loss = mse(computed, valLabels)
			print(epoch, loss.item(), file = ext2, sep = ",")

		if hparams.epochTrainingR2:
			#COMPUTE METRICS
			trainOut = net(trainingData)
			computed = torch.select(trainOut,1,0) #changes shape from (valData.size, 1) to (valData.size)

			#REPORT METRICS
			r2score = tm.R2Score()
			r2 = r2score(computed, trainingLabels)
			print(epoch, r2.item(), file = ext3, sep = ",")

		if hparams.epochValR2:
			#COMPUTE METRICS
			valOut = net(valData)
			computed = torch.select(valOut,1,0) #changes shape from (valData.size, 1) to (valData.size)

			#REPORT METRICS
			r2score = tm.R2Score()
			r2 = r2score(computed, valLabels)
			print(epoch, r2.item(), file = ext4, sep = ",")

		#early stopping
		if epoch == 8:
			currentBestValidationLoss = mseLoss(net, valData, valLabels)
			bestNet = net

		if epoch > 8 and (epoch-3)%5 == 0:
			currentValidationLoss = mseLoss(net, valData, valLabels)

			if currentValidationLoss > currentBestValidationLoss:
				return bestNet

			currentBestValidationLoss = currentValidationLoss
			bestNet = net

		epoch += 1
	return bestNet

def testModel(net, testData, testLabels, twoDScaler, trainingLabelsMean, trainingLabelsStdev, ext5):
	#SETTING UP DATA
	data = testData
	twoDScaler.transform(data)

	#RUN THROUGH NETWORK
	out = net(data)
	computed = torch.select(out, 1, 0)

	#COMPUTE METRICS
	computed = computed * trainingLabelsStdev + trainingLabelsMean
	mse = nn.MSELoss()
	loss = mse(computed, testLabels)

	#OUTPUT
	if hparams.tgas_actual_predicted:
		for i in range(10000):
			print(round(testLabels[i].item(), 3), round(computed[i].item(), 3), sep = ",", file = ext5)
	
	return loss.item()

def mseLoss(net, valData, valLabels):
	#SETTING UP DATA
	data = valData
	target = valLabels

	#COMPUTE METRICS
	out = net(data)
	criterion = nn.MSELoss()
	computed = torch.select(out, 1, 0)
	loss = criterion(computed, target)
	
	return loss