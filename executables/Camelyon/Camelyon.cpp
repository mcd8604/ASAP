#include <string>
#include <vector>
#include <thread>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
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

using namespace boost;
using namespace program_options;
using namespace filesystem;
using namespace std;
using namespace pathology;
using namespace cv;

int main(int argc, char *argv[]) {
	string imageFilePath;
	bool train;
	string rfModel;
	string groundTruthXML;
	string groundTruthMask;
	string testResultFile;

	options_description optionsDesc("Options");
	optionsDesc.add_options()
		("help,h", "Displays this message")
		("inputImage,i", value<std::string>(&imageFilePath)->required(), "Path to input image TIF")
		("train,t", value<bool>(&train)->default_value(true), "Set whether to train a model and save it or load a model and test it.")
		("randomForestModel,f", value<string>(&rfModel)->default_value("rf.yaml"), "Set the file path for the random forest model YAML")
		("groundTruthXML,x", value<string>(&groundTruthXML)->default_value(""), "Set the file path for the ground truth XML annotations.")
		("groundTruthMask,m", value<string>(&groundTruthMask)->default_value(""), "Set the file path for the ground truth TIF mask")
		("testResultFile,r", value<string>(&testResultFile)->default_value("result.yaml"), "Set the file path for the test result output (XML or YAML).")
		;

	variables_map vm;
	try {
		store(command_line_parser(argc, argv).options(optionsDesc).run(), vm);
		notify(vm);
		if (vm.count("help")) {
			std::cout << optionsDesc <<endl;
			return 0;
		}
	}
	catch (required_option& e) {
		cerr << "ERROR: " << e.what() <<endl <<endl;
		cerr << "Use -h or --help for usage information" <<endl;
		return -1;
	}

	if (!exists(imageFilePath)) {
		cerr << "ERROR: Invalid input image: " << imageFilePath << endl;
		return -1;
	}
	MultiResolutionImageReader reader; 
	MultiResolutionImage* input = reader.open(imageFilePath);
	MultiResolutionImage *groundTruthImage;

	if (input) {
		Slide *slide;
		if (groundTruthXML != "") {
			if (!exists(groundTruthXML)) {
				cerr << "ERROR: Invalid ground truth annotation file: " << groundTruthXML << endl;
				return -1;
			}
			AnnotationService annoSvc;
			if (annoSvc.loadRepositoryFromFile(groundTruthXML)) {
				slide = new Slide(input, annoSvc.getList());
			}
			else {
				cerr << "ERROR: Could not load annotation file: " << groundTruthXML << endl;
				return -1;
			}
		}
		else if(groundTruthMask != "") {
			if (!exists(groundTruthMask)) {
				cerr << "ERROR: Invalid ground truth mask file: " << groundTruthMask << endl;
				return -1;
			}
			groundTruthImage = reader.open(groundTruthMask);
			if (!groundTruthImage) {
				cerr << "ERROR: Invalid ground truth mask file: " << groundTruthMask << endl;
				return -1;
			}
			slide = new Slide(input, groundTruthImage);
		}

		//train or test the random forest
		if(train) {
			slide->rfTrain(rfModel);
		} else {
			Mat testResult = slide->rfTest(rfModel, testResultFile);
		}

		delete slide;
		delete input;
		delete groundTruthImage;
    }
    else {
     cerr << "ERROR: Invalid input image" <<endl;
    }
	return 0;
}

