
#include "LBPFeatureStrategy.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// This strategy is based on the following paper:
//
// Ojala, T.; Pietikainen, M.; Maenpaa, T., 
// "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," 
// Pattern Analysis and Machine Intelligence, IEEE Transactions on, 
// vol.24, no.7, pp.971,987, Jul 2002

void LBPFeatureStrategy::init() {
	//mFeatureNames(mNumBins);
	for (int i = 0; i < getNumFeatures(); i++)
		mFeatureNames.push_back("bin" + std::to_string(i));
	mFilter = new ColorDeconvolutionFilter<double>();
}

unsigned char LBPFeatureStrategy::getLBPUniformVal(const bool lbp[8]) {
	int numTransitions = 0;
	int numOnes = lbp[0] ? 1 : 0;
	for (int i = 0; i < 7; i++) {
		if (lbp[i] != lbp[i + 1]) ++numTransitions;
		if (lbp[i + 1]) ++numOnes;
	}
	if (lbp[0] != lbp[7]) ++numTransitions;
	unsigned char uniformVal = numTransitions <= 2 ? numOnes : 9;
	return uniformVal;
}

std::vector<float> LBPFeatureStrategy::constructFeatures(Patch<double> &tilePatch) {
	std::vector<float> featureVector(getNumFeatures(), 0);

	Patch<double> hemaPatch;
	mFilter->filter(tilePatch, hemaPatch);
	cv::Mat_<double> hemaMat = patchToMat(hemaPatch);
	
	cv::Mat_<unsigned char> lbpMat = cv::Mat_<unsigned char>::zeros(hemaMat.size());
	for (int y = 1; y < lbpMat.rows - 1; y++) {
		for (int x = 1; x < lbpMat.cols - 1; x++) {
			double center = hemaMat.at<double>(y, x);
			if (!mIsUniform) {
				unsigned char lbp = 0;
				lbp |= (hemaMat.at<double>(y - 1, x + 0) > center) << 7;
				lbp |= (hemaMat.at<double>(y - 1, x + 1) > center) << 6;
				lbp |= (hemaMat.at<double>(y + 0, x + 1) > center) << 5;
				lbp |= (hemaMat.at<double>(y + 1, x + 1) > center) << 4;
				lbp |= (hemaMat.at<double>(y + 1, x + 0) > center) << 3;
				lbp |= (hemaMat.at<double>(y + 1, x - 1) > center) << 2;
				lbp |= (hemaMat.at<double>(y + 0, x - 1) > center) << 1;
				lbp |= (hemaMat.at<double>(y - 1, x - 1) > center) << 0;
				lbpMat.at<unsigned char>(y, x) = lbp;
			} else {
				bool lbp[8] = {
					hemaMat.at<double>(y - 1, x + 0) > center,
					hemaMat.at<double>(y - 1, x + 1) > center,
					hemaMat.at<double>(y + 0, x + 1) > center,
					hemaMat.at<double>(y + 1, x + 1) > center,
					hemaMat.at<double>(y + 1, x + 0) > center,
					hemaMat.at<double>(y + 1, x - 1) > center,
					hemaMat.at<double>(y + 0, x - 1) > center,
					hemaMat.at<double>(y - 1, x - 1) > center
				};
				lbpMat.at<unsigned char>(y, x) = getLBPUniformVal(lbp);
			}
		}
	}

	//cv::imshow("hemaMat", hemaMat);
	//cv::imshow("lbp", lbp);
	//cv::waitKey();

	// histogram
	if (mIsUniform) {
		double hist[10] = { 0 };
		cv::MatIterator_<uchar> i = lbpMat.begin();
		cv::MatIterator_<uchar> end = lbpMat.end();
		for (; i != end; ++i)
			++hist[*i];
		// normalize
		double numPixels = lbpMat.size().area();
		for (int b = 0; b < 10; ++b)
			featureVector[b] = hist[b] / numPixels;
	} else {
		cv::Mat hist;
		int histSize = 256;
		float range[2] = { 0, 255 };
		const float* ranges[] = { range };
		calcHist(&lbpMat, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false);
		for (int b = 0; b < histSize; b++)
			featureVector[b] = hist.at<float>(b, 0);
	}
	return featureVector;
}