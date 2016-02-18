#include <string>
#include <vector>

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
using namespace ml;

int main(int argc, char *argv[]) {
	if (argc < 3) {
		cerr << "Usage: Camelyon imageFilePath annotationFilePath";
		return -1;
	}
	string imageFilePath = argv[1];
	string annotationFilePath = argv[2];

	MultiResolutionImageReader reader; 
	MultiResolutionImage* input = reader.open(imageFilePath);

	if (input) {
		Slide *slide = new Slide(input);
		AnnotationService annoSvc;
		if (annoSvc.loadRepositoryFromFile(annotationFilePath)) {
			slide->setAnnotationList(annoSvc.getList());
		}

		/// Pre-processing - Tissue Classification
		// Native (level 0) slide dimensions happen to be multiples of 512
		vector<Rect> tiles = slide->getTissueTiles(8, Size(512, 512));

		/// Pre-processing - Superpixel Segmentation
		// TODO generate superpixels on each tile at native resolution

		/// Feature construction
		Mat features = slide->constructFeatures({ SimpleBlobDetector::create(), GFTTDetector::create(), ORB::create() }, tiles, 0);

		/// Feature selection
		Ptr<TrainData> trainData = TrainData::create(features, SampleTypes::ROW_SAMPLE, slide->generateGroundTruth(tiles));

		// TODO SVM
		Ptr<SVM> svm = SVM::create();
		svm->trainAuto(trainData);

		// TODO RF

		// TODO Evaluate results

		delete slide;
		delete input;
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
	return 0;
}

