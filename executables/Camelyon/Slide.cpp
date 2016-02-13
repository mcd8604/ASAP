#include <vector>
#include "Slide.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

Slide::Slide(MultiResolutionImage * image, SuperpixelType sType)
{
	mImage = image;
	mSType = sType;
}

Slide::~Slide()
{
}

// TODO this should really be in FeatureExtractor class
bool Slide::segFeatsToCSV(std::string filePath)
{
	// temporary - for now just throw it in the console
	for (Segment s : segments) {
		cout << to_string(s.getId()) + ": ";
		for(double f : s.getFeatures()) {
			cout << to_string(f) + ", "; // TODO handle edge case
		}
		cout << "\n";
	}

	return false;
}

void Slide::evaluatePredictions()
{
}

bool isForeground(Mat *img, Rect r, double densityThresh, double backgroundThresh) {
	double count = 0.0;
	for (int y = 0; y < r.height; y++)
		for (int x = 0; x < r.width; x++)
			if (img->at<double>(Point(r.x + x, r.y + y)) > densityThresh)
				count++;
	return (count / (r.width * r.height)) > backgroundThresh;
}

// TODO refactor process out of Slide class?
// TODO determine if process generalization is necessary. 
//		can we just stick with processing one low res level and the native level?
void Slide::constructFeatures(Ptr<Feature2D> featureDetector) {
	// TODO parameterize low res level
	int lowResLvl = 7;
	CV_Assert(lowResLvl < mImage->getNumberOfLevels());
	vector<unsigned long long, allocator<unsigned long long>> lowResDim = mImage->getLevelDimensions(lowResLvl);
	Patch<uchar> lowResPatch = mImage->getPatch<uchar>(0, 0, lowResDim[0], lowResDim[1], lowResLvl);

	// (temporary check)
	Mat lowResMat = patchToMat(lowResPatch);
	cvtColor(lowResMat, lowResMat, COLOR_BGR2HSV);
	imshow("HSV", lowResMat);

	// Convert low res image to HSD color model
	// TODO adjust filter levels - minor non-tissue artifacts are present
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	//filter->setGlobalDensityThreshold(0.0);
	Patch<double> hemaPatch, eosinPatch;
	filter->filter(lowResPatch, hemaPatch);
	filter->setOutputStain(1);
	filter->filter(lowResPatch, eosinPatch);
	Mat hema = patchToMat(hemaPatch);
	Mat eosin = patchToMat(eosinPatch);

	// (temporary check)
	imshow("hema", hema);
	imshow("eosin", eosin);
	waitKey(0);

	// TODO parameterize tile size
	int tileWidth = 512;
	int tileHeight = 512;
	vector<unsigned long long, allocator<unsigned long long>> dim = mImage->getLevelDimensions(0);
	float numTilesX = float(dim[0]) / tileWidth;
	float numTilesY = float(dim[1]) / tileHeight;

	// Assert tile size always fits integrally
	CV_Assert(numTilesX == (int)numTilesX);
	CV_Assert(numTilesY == (int)numTilesY);

	int d = mImage->getLevelDownsample(lowResLvl);
	int lowResTileWidth = lowResDim[0] / numTilesX;
	int lowResTileHeight = lowResDim[1] / numTilesY;

	// Assert downsampling
	//CV_Assert(lowResTileWidth * numTilesX == dim[0]);
	//CV_Assert(lowResTileHeight * numTilesY == dim[1]);

	// Iterate over tiles 
	if (mSType == SuperpixelType::SLIC) {

		// TODO integrate SLIC, add contour points

	}
	else if (mSType == SuperpixelType::RECT) {
		// TODO this function is getting too long..
		for (int y = 0; y < numTilesY; y++) {
			for (int x = 0; x < numTilesX; x++) {

				int left = x * lowResTileWidth;
				int top = y * lowResTileHeight;
				int right = left + lowResTileWidth;
				int bottom = top + lowResTileHeight;
				Point topLeft = Point(left, top);
				Point bottomRight = Point(right, bottom);
				Rect r(topLeft, bottomRight);

				if (isForeground(&hema, r, 0.2, 0.1)) {
					// Get level 0 for tile
					Patch<uchar> p = mImage->getPatch<uchar>(x * tileWidth, y * tileHeight, tileWidth, tileHeight, 0);
					Mat m = patchToMat(p);
					Segment s(y * numTilesX + x);
					vector<Point> points;
						// Clockwise starting at top left
						points.resize(4);
						points[0] = topLeft;
						points[1] = Point(top, right);
						points[2] = bottomRight;
						points[3] = Point(bottom, left);
					s.setContour(points);
					s.constructFeatures(m, featureDetector);
					segments.push_back(s);
					//cout << "Constructed Features for (" << x << "," << y << ")\n";
				}
			}
		}
	}

	delete filter;
}