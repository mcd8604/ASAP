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
		Slide *slide = new Slide();
		AnnotationService annoSvc;
		if (annoSvc.loadRepositoryFromFile(annotationFilePath)) {
			shared_ptr<AnnotationList> annoList = annoSvc.getList();
			slide->setAnnotationList(annoList);
		}

		// Native (level 0) slide dimensions happen to be multiples of 512
		vector<Rect> lvl8Rects = slide->getTissueTiles(input, 8, Size(512, 512));

		// TODO generate superpixels on each tile at native resolution

		// TODO Feature construction
		//Ptr<SimpleBlobDetector> featureDetector = SimpleBlobDetector::create();
		
		// TODO Feature selection (SVM/RF)

		// TODO Evaluate results

		delete slide;
		delete input;
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
	return 0;
}

