#include "neuralNteworks.h"


void readImages(string imagesPath, vector<string> &ImageSeries)
{

	//Read images
	char search_path[200];
	sprintf(search_path, "%s*.*", imagesPath.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				string imagePath_ = imagesPath + fd.cFileName;
				std::size_t found = imagePath_.find(".bmp");
				if (found == std::string::npos)
				{
					found = imagePath_.find(".png");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".PNG");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".BMP");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".JPG");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".jpg");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".JPEG");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".jpeg");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".JPEG");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".TIFF");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".tiff");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".tif");
				}
				if (found == std::string::npos)
				{
					found = imagePath_.find(".TIF");
				}
				if (found == std::string::npos)
				{
					continue;
				}

				ImageSeries.push_back(imagePath_);


			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);

	}


}

int main(int* argc, char** argv)
{

	//int a[3] = {1,2,3};
	//int* p = &a[0];
	//int**q = &p;
	//cout << "Data" << "\t" << *((*q)+1);

	//cout << "*****************Feature Extraction & Learning **********************\n \n" << endl;

	//cout << "Reading Images \n" << endl;
	///*Read the postive & negative test and train images*/
	//string positiveTrainPath = "D:\\train\\analytics\\frontNew\\";//"D:\\train\\analytics\\posLift\\"; D:\\train\\\closed
	//vector<string>posImagesTrain;
	//readImages(positiveTrainPath, posImagesTrain);
	//cout << "\n Total Positive Images for Train:: " << posImagesTrain.size() << endl;
	//string negetiveTrainPath = "D:\\train\\analytics\\halfBodyNeg\\"; vector<string>negImagesTrain;
	//readImages(negetiveTrainPath, negImagesTrain);
	//cout << "\n Total Negative Images for Train:: " << negImagesTrain.size() << endl;

	///********Read the images to Mat ****/
	//int posImageCountTrain = posImagesTrain.size();
	//int negImageCountTrain = negImagesTrain.size();
	//int totalImageCountTrain = posImageCountTrain + negImageCountTrain;

	//y = sin(x);
	Mat x(1, 63, CV_32FC1);
	Mat y(1, 63, CV_32FC1);
	int itr = 0;
	for (float i = -3.14 / 2; i <= 3.14/2; i = i + 0.1)
	{
		y.at<float>(0, itr) = (int(((sin(i) + 1.0) / 2.0)*100) /100.0);
		x.at<float>(0, itr) = (int((i/3.14)*2000)/1000.0);
		cout.precision(3);
		cout << "X: " << std::fixed  << x.at<float>(0, itr)
			<< "\t Y: " << std::fixed <<  y.at<float>(0, itr) << endl;
		itr++;
	}

	int noOflayers = 3;
	int noOfNeuronsPerLayer[3] = { 1, 4, 1 };
	
	neuralNetworks NN(noOflayers, noOfNeuronsPerLayer);
	
	NN.trainNetwork(x, y, true, 63);

	Mat testSample(1024, 1, CV_32FC1);
	testSample = 1;

	NN.predictNetworkResponce(testSample);

	return 0;
}
