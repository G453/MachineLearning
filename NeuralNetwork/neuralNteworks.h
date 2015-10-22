#ifndef __NEURALNETWORKS_HPP_INCLUDED
#define __NEURALNETWORKS_HPP_INCLUDED
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>

#include <direct.h>
#include <Windows.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;

class neuralNetworks
{
public:
	neuralNetworks(int , int* );
	~neuralNetworks();
	Mat predictNetworkResponce(Mat& inputSample);
	void trainNetwork(Mat &inputSamples, Mat &classLabels,bool batchGradient, int bactchSize);
private:
	vector<Mat> weights;
	vector<Mat> aInEachLayer;//Non Linear funciton outputs for first layer it is just input
	vector<Mat> zInEachLayer;//Sum of previous layer inputs * weights
	vector<Mat> deltaInEachLayer;
	int noOfLayers;
	int noOfHiddenLayers;
	vector<int> noOfNeuransPerLayer;
	int noOfSamples;
	void initializeActivations(int batchSize);
	void forwardPass(Mat& );
	Mat createRandomMatrix(int cols, int rows);
	Mat nonLinearity(Mat);
	
};



#endif