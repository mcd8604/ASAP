#include <string>
#include <vector>
#include <fstream>

#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "AnnotationService.h"
#include "AnnotationList.h"
#include "core/filetools.h"
#include "config/pathology_config.h"
#include "Slide.h"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace pathology;
using namespace cv;

int main(int argc, char *argv[]) {
	if (argc < 3) {
		cerr << "Usage: Camelyon imageFilePath annotationFilePath";
		return -1;
	}
	string imageFilePath = argv[1];
	string groundTruthFilePath = argv[2];

	MultiResolutionImageReader reader; 
	MultiResolutionImage* input = reader.open(imageFilePath);

	if (input) {
		Slide *slide = new Slide(input);
		MultiResolutionImage *groundTruthImage = reader.open(groundTruthFilePath);
		slide->setGroundTruth(groundTruthImage);
		/*AnnotationService annoSvc;
		if (annoSvc.loadRepositoryFromFile(annotationFilePath)) {
			slide->setAnnotationList(annoSvc.getList());
		}*/

		/// Pre-processing - Tissue Classification
		// Native (level 0) slide dimensions happen to be multiples of 512
		slide->classifyTissueTiles(8, Size(512, 512));

		/// Pre-processing - Superpixel Segmentation
		// TODO generate superpixels on each tile at native resolution

		/// Feature construction
		int runLevel = 4;
		vector<Rect> tiles = slide->getTissueTiles(runLevel);
		//Mat features = slide->constructFeatures({ SimpleBlobDetector::create(), GFTTDetector::create(), ORB::create() }, tiles, runLevel);
		vector<string> featureNames;
		Mat features = slide->constructFeatures(tiles, runLevel, &featureNames);
		Mat groundTruth = slide->getGroundTruth(tiles, runLevel);

		/// Feature selection
		// Currently just storing to files for testing
		ofstream csv("features.csv");
		for (int f = 0; f < featureNames.size(); f++) {
			if (f > 0) csv << ",";
			csv << featureNames[f];
		}
		csv << ",ground_truth";
		for (int x = 0; x < features.rows; x++) {
			csv << "\n";
			for (int y = 0; y < features.cols; y++) {
				if (y > 0) csv << ",";
				csv << features.at<float>(x, y);
			}
			csv << "," << groundTruth.at<float>(x, 0);
		}
		csv.close();

		// TODO SVM
		//Ptr<SVM> svm = SVM::create();
		//svm->trainAuto(trainData);
		//Mat resp;
		//float pct = svm->calcError(trainData, true, resp);
		
		// TODO RF
		slide->rforest(groundTruth, features);

		/// Classification

		/// TODO Evaluate results

		delete slide;
		delete input;
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
	return 0;
}

