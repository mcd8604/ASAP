#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "AnnotationService.h"
#include "AnnotationList.h"
#include "core/filetools.h"
#include "config/pathology_config.h"
#include "Slide.h"
#include "opencv2/features2d.hpp"

#include "Camelyon.h"

using namespace std;
using namespace pathology;
using namespace cv;

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
		Slide *slide = new Slide(input, SuperpixelType::RECT);
		AnnotationService annoSvc;
		if (annoSvc.loadRepositoryFromFile(annotationFilePath)) {
			shared_ptr<AnnotationList> annoList = annoSvc.getList();
			slide->setAnnotationList(annoList);
		}

		Ptr<SimpleBlobDetector> featureDetector = SimpleBlobDetector::create();
		slide->constructFeatures(featureDetector);
		
		// Output results to csv
		slide->segFeatsToCSV("features.csv");

		// TODO Train/Test model

		// TODO Evaluate results
		slide->evaluatePredictions();

		delete slide;
		delete input;
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
	return 0;
}

