#include "neuralNteworks.h"



float sigmoidDerivative(float z)
{
	return z *(1 - z);
}

Mat sigmoidDerivative(Mat z)
{
	z.convertTo(z, CV_32FC1);
	Mat ones(z.rows, z.cols, CV_32FC1);
	ones = 1;
	Mat diff_ = ones - z;
	ones = z.mul(diff_);
	return ones;
}

float sigmoidNonLinearity(float z)
{
	return (1.0 / (1.0 + exp(-z)));
}

Mat addBiasFeatureToInput(Mat& inputVector)
{
	Mat biasFeatueAddedInput;
	biasFeatueAddedInput.create(inputVector.rows + 1, inputVector.cols, CV_32FC1);
	biasFeatueAddedInput = 1;
	inputVector.copyTo(biasFeatueAddedInput(Rect(0, 0, inputVector.cols, inputVector.rows)));
	return biasFeatueAddedInput;

}


Mat neuralNetworks::createRandomMatrix(int rows, int cols)
{
	Mat randomMat(rows, cols, CV_32FC1);
	randomMat = 0;

	for (int itrCols = 0; itrCols < randomMat.cols; itrCols++)
	{
		for (int itrRows = 0; itrRows < randomMat.rows; itrRows++)
		{
			randomMat.at<float>(itrRows, itrCols) = (rand() % 100) * 0.001;
		}
	}
	return randomMat;
}

neuralNetworks::neuralNetworks(int nLayers, int* noOfNuerons)
{
	noOfLayers = nLayers;
	noOfHiddenLayers = noOfLayers - 2;
	/*Add the bias neuran in each layer except at outputlayer*/
	for (int itrLayers = 0; itrLayers < noOfLayers; itrLayers++)
	{
		if (itrLayers < noOfLayers - 1)
			noOfNeuransPerLayer.push_back(noOfNuerons[itrLayers]+1);
		else
			noOfNeuransPerLayer.push_back(noOfNuerons[itrLayers]);
	}
	/*Create random weight vectors in each layer*/

	for (int itrLayers = 0; itrLayers < noOfHiddenLayers + 1; itrLayers++)
	{
		if (itrLayers < noOfHiddenLayers)
		{
			Mat weightsPerLayer = createRandomMatrix(noOfNeuransPerLayer[itrLayers + 1] - 1, noOfNeuransPerLayer[itrLayers]);
			weights.push_back(weightsPerLayer);
		}
		else
		{
			Mat weightsPerLayer = createRandomMatrix(noOfNeuransPerLayer[itrLayers + 1], noOfNeuransPerLayer[itrLayers]);
			weights.push_back(weightsPerLayer);
		}
	}
	
}

void neuralNetworks::forwardPass(Mat& inputSample)
{
	int noOfSamplesPerBatch = inputSample.cols;
	Mat activationsInputLayer = addBiasFeatureToInput(inputSample);
	/*copy the persent activations to activations list*/
	aInEachLayer.push_back(activationsInputLayer);

	Mat activationsOutputLayer(noOfNeuransPerLayer[noOfLayers - 1], noOfSamplesPerBatch, CV_32FC1);
	for (int itrLayers = 0; itrLayers < weights.size(); itrLayers++)
	{

		Mat weightsPerLayer = weights[itrLayers];
		Mat zNextLayer = weightsPerLayer * activationsInputLayer;
		Mat aNextLayer = nonLinearity(zNextLayer);

		//release the previous data
		activationsInputLayer.release();

		activationsInputLayer.create(aNextLayer.rows + 1, aNextLayer.cols, CV_32FC1);
		activationsInputLayer = 1;
		aNextLayer.copyTo(activationsInputLayer(Rect(0, 0, aNextLayer.cols, aNextLayer.rows)));
		
		//Copy the data for BP
		zInEachLayer.push_back(zNextLayer);
		aInEachLayer.push_back(aNextLayer);

		if (itrLayers == weights.size() - 1)
			aNextLayer.copyTo(activationsOutputLayer);

		//Release the old data
		weightsPerLayer.release();
		zNextLayer.release();
		aNextLayer.release();
	}
	return ;
}

void neuralNetworks::trainNetwork(Mat &inputSamples, Mat &classLabels, bool batchGradient, int bactchSize)
{
	/*Forward Pass*/
	Mat batchInput = inputSamples(Rect(0, 0, bactchSize, inputSamples.rows));
	batchInput.convertTo(batchInput, CV_32FC1);
	Mat batchLabels = classLabels(Rect(0, 0, bactchSize, classLabels.rows));
	batchLabels.convertTo(batchLabels, CV_32FC1);
	forwardPass(batchInput);
	/*Compute the Delta for each layer*/
	Mat deltaOutputLayer = -1.0 * (batchLabels - aInEachLayer.back()).mul(sigmoidDerivative(zInEachLayer.back()));
	Mat deltaInNextLayer;
	deltaOutputLayer.copyTo(deltaInNextLayer);
	deltaInEachLayer.push_back(deltaInNextLayer);
	for (int itrLayers = weights.size(); itrLayers >0; itrLayers++)
	{
		Mat weightTranspose;
		transpose(weights[itrLayers - 1], weightTranspose);
		Mat deltaInPresentLayer = (weightTranspose * deltaInNextLayer).mul(sigmoidDerivative(zInEachLayer[itrLayers - 1]));
		deltaInNextLayer.release();
		deltaInPresentLayer.copyTo(deltaInNextLayer);
		deltaInEachLayer.push_back(deltaInPresentLayer);
	}
	/*Delta Mats are stored in Reverse so will acees them in reverse order*/

	/*Update the weights according to the updates*/
	vector<Mat> updatedWeights;
	int itrEnd = deltaInEachLayer.size()-1;
	for (int itrLayers = 0; itrLayers < weights.size(); itrLayers++)
	{
		Mat activationsTranspose;
		transpose(aInEachLayer[itrLayers], activationsTranspose);
		Mat updatedWeightPerLayer = weights[itrLayers] + deltaInEachLayer[itrEnd] * activationsTranspose;
		updatedWeights.push_back(updatedWeightPerLayer);
		itrEnd--;
	}
	/*replace the old weights with updated*/
	weights.clear();
	for (int itrWeights = 0; itrWeights < updatedWeights.size(); itrWeights++)
	{
		weights.push_back(updatedWeights[itrWeights]);
	}

}

Mat neuralNetworks::nonLinearity(Mat z)
{
	Mat a(z.rows, z.cols, CV_32FC1);

	for (int iRows = 0; iRows < z.rows; iRows++)
	{
		for (int iCols = 0; iCols < z.cols; iCols++)
		{
			a.at<float>(iRows, iCols) = sigmoidNonLinearity(a.at<float>(iRows, iCols));
		}
	}
	return a;
}

Mat neuralNetworks::predictNetworkResponce(Mat& inputSample)
{
	noOfSamples = inputSample.cols;
	Mat activationsInputLayer = addBiasFeatureToInput(inputSample);
	
	Mat activationsOutputLayer(noOfNeuransPerLayer[noOfLayers-1], noOfSamples, CV_32FC1);
	for (int itrLayers = 0; itrLayers < weights.size(); itrLayers++)
	{
		
		Mat weightsPerLayer = weights[itrLayers];
		Mat zNextLayer = weightsPerLayer * activationsInputLayer;
		Mat aNextLayer = nonLinearity(zNextLayer);
		//release the previous data
		activationsInputLayer.release();
		weightsPerLayer.release();
		zNextLayer.release();
		activationsInputLayer.create(aNextLayer.rows+1, aNextLayer.cols, CV_32FC1);
		activationsInputLayer = 1;
		aNextLayer.copyTo(activationsInputLayer(Rect(0,0,aNextLayer.cols,aNextLayer.rows)));
		if (itrLayers == weights.size() - 1)
			aNextLayer.copyTo(activationsOutputLayer);
		aNextLayer.release();
	}
	return activationsOutputLayer;
}

neuralNetworks::~neuralNetworks()
{
}