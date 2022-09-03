#HYPERPARAMETERS
numEpochs = 200
batchSize = 10000
learningRate = 0.002

#FILES
outputFile = "out.txt"
outputFile2 = "out.txt"
outputFile3 = "out.txt"
outputFile4 = "out.txt"
outputFile5 = "out.txt"

datasetFile = "tgas_constrained.pkl"

#NETWORK SETTINGS
crossVal = True
proportionTrainingDatasetUsed = 1.0

#OUTPUT SETTINGS
epochTrainingLoss = False #MSE loss on training data after each epoch. outputFile
epochValLoss = False #MSE loss on validation data after each epoch. outputFile2
epochTrainingR2 = False #R2 score from training data after each epoch. outputFile3
epochValR2 = False #compute model's R2 score from validation data after each epoch. outputFile4

tgas_actual_predicted = False # actual and predicted gas temps for random points. outputFile5
modelTesting = False #evaluates model on test set

saveModel = False #saves model of first fold if crossvalidation used
modelFile = "model.pth"



