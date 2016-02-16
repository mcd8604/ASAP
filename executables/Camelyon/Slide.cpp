#include <vector>
#include "Slide.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace ml;

Slide::Slide(MultiResolutionImage * image)
{
	mImage = image;
}

Slide::~Slide()
{
}

void Slide::setAnnotationList(shared_ptr<AnnotationList> annoList)
{
	mAnnoList = annoList;
}

// Uses the given image and level to classify tissue tiles and returns them as rectangles of size nativeTileSize 
vector<Rect> Slide::getTissueTiles(int sampleLevel, Size targetTileSize) {
	vector<Rect> tissueTiles;

	CV_Assert(sampleLevel < mImage->getNumberOfLevels());
	vector<unsigned long long, allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(sampleLevel);
	Patch<uchar> levelPatch = mImage->getPatch<uchar>(0, 0, levelDim[0], levelDim[1], sampleLevel);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	// TODO adjust filter levels - minor non-tissue artifacts are present
	//filter->setGlobalDensityThreshold(0.25);
	Patch<double> hemaPatch;
	filter->filter(levelPatch, hemaPatch);
	Mat hema = patchToMat(hemaPatch);
	delete filter;

	double downsample = mImage->getLevelDownsample(sampleLevel);
	int w = targetTileSize.width / downsample;
	int h = targetTileSize.height / downsample;
	int numTilesX = levelDim[0] / w;
	int numTilesY = levelDim[1] / h;

	for (int y = 0; y < numTilesY; y++) {
		for (int x = 0; x < numTilesX; x++) {
			Rect r(x * w, y * h, w, h);
			// Classify Foreground/Background
			// The hema patch has binary values 
			// (ok, apparently not..)
			// TODO investigate ColorDeconFilter
			// so this yields the percent of foreground pixels
			Scalar sc = sum(hema(r));
			double c1 = sc[0];
			double ar = r.area();
			double percentForeground = c1 / ar;
			if (percentForeground > 0.1) {
				tissueTiles.push_back(Rect(Point(x * targetTileSize.width, y * targetTileSize.height), targetTileSize));
			}
		}
	}
	return tissueTiles;
}

#define TEST_FC_VIEW_TILES 0;
OutputArray Slide::constructFeatures(vector<Ptr<Feature2D>> featureDetectors, vector<Rect> tiles, int level) {
	Mat m;
	int numTiles = tiles.size();
	int numFeatureDetectors = featureDetectors.size();
	// TODO parameterize statistics
	const int numStats = 1;
	Mat features(numTiles, numFeatureDetectors * numStats, CV_32FC1);
	vector<KeyPoint> keyPoints;
	//Mat descriptors;
	for (int i = 0; i < numTiles; i++) {
		Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, level);
		m = patchToMat(p);
		for (Ptr<Feature2D> fd : featureDetectors) {
			//fd->clear();
			fd->detect(m, keyPoints);

			// TODO parameterize statistics
			float count = 0;
			float sumSize = 0;			
			for (KeyPoint keyPoint : keyPoints) {
				count++;
				sumSize += keyPoint.size;
			}
			float meanSize = sumSize / count;
			
			features.at<float>(i, 0) = count;
			features.at<float>(i, 1) = sumSize;
			features.at<float>(i, 2) = meanSize;

			//fd->detectAndCompute(m, noArray(), keyPoints, descriptors);
			//features.push_back(descriptors);

#if TEST_FC_VIEW_TILES
			if(i >= 500) {
				drawKeypoints(m, keyPoints, m);
				imshow("test", m);
				//imshow("descriptors", descriptors);
				waitKey(0);
			}
#endif
			// TODO feature reduction
		}
	}
	return features;
}