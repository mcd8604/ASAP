
#include "LBPFeatureStrategy.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

void LBPFeatureStrategy::init() {
	//mFeatureNames(mNumBins);
	for (int i = 0; i < mNumBins; i++)
		mFeatureNames.push_back("bin" + std::to_string(i));
	mFilter = new ColorDeconvolutionFilter<double>();
}

std::vector<float> LBPFeatureStrategy::constructFeatures(Patch<double> tilePatch) {
	std::vector<float> featureVector(0., mNumBins);

	Patch<double> hemaPatch;
	mFilter->filter(tilePatch, hemaPatch);
	cv::Mat_<double> hemaMat = patchToMat(hemaPatch);

	// Local Binary Pattern
	/*
	Copyright (c) 2011, philipp <bytefish[at]gmx.de>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
	* Neither the name of the organization nor the
	names of its contributors may be used to endorse or promote products
	derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	*/

	cv::Mat_<unsigned char> lbp = cv::Mat_<unsigned char>::zeros(hemaMat.size());
	for (int k = 1; k < lbp.rows - 1; k++) {
		for (int j = 1; j < lbp.cols - 1; j++) {
			double center = hemaMat.at<double>(k, j);
			unsigned char code = 0;
			code |= (hemaMat.at<double>(k - 1, j - 1) >= center) << 7;
			code |= (hemaMat.at<double>(k - 1, j) >= center) << 6;
			code |= (hemaMat.at<double>(k - 1, j + 1) >= center) << 5;
			code |= (hemaMat.at<double>(k, j + 1) >= center) << 4;
			code |= (hemaMat.at<double>(k + 1, j + 1) >= center) << 3;
			code |= (hemaMat.at<double>(k + 1, j) >= center) << 2;
			code |= (hemaMat.at<double>(k + 1, j - 1) >= center) << 1;
			code |= (hemaMat.at<double>(k, j - 1) >= center) << 0;
			lbp.at<unsigned char>(k, j) = code;
		}
	}

	//cv::imshow("hemaMat", hemaMat);
	//cv::imshow("lbp", lbp);
	//cv::waitKey();

	// histogram
	cv::Mat hist;
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* ranges[] = { range };
	calcHist(&lbp, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false);
	for (int b = 0; b < histSize; b++)
		featureVector[b] = hist.at<float>(b, 0);
	return featureVector;
}