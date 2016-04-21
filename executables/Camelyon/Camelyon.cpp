#include <string>

#include "config/pathology_config.h"
#include "FeatureConstructor.h"
#include "LBPFeatureStrategy.h"
#include "opencv2/ml.hpp"
#include "TestResults.h"

#include "ModelTrainer.h"
#include "ModelTester.h"


#include "opencv2\highgui.hpp"
#include "boost\filesystem.hpp"
#include "boost\algorithm\string.hpp"

using namespace boost;
using namespace boost::filesystem;

using namespace std;
using namespace cv;

void test(std::string imageDir, std::string modelFilePath) {
	ModelTester tester;
	tester.loadSVMModel(modelFilePath);
	TestResults testResults = tester.Test(imageDir);
	testResults.plotROC(1000);
}

void constructAll() {
	string dir = "C:/CAMELYON_TRAIN_DATA/";
	string fileNames[19] = {
		//"Tumor_001",
		//"Tumor_002",
		//"Tumor_003",
		//"Tumor_004",
		//"Tumor_005",
		//"Tumor_006",
		//"Tumor_007",
		//"Tumor_008",
		//"Tumor_009",
		//"Tumor_010",
		//"Tumor_011",
		//"Tumor_012",
		//"Tumor_013",
		//"Tumor_014",
		//"Tumor_015",
		//"Tumor_016",
		//"Tumor_017",
		//"Tumor_018",
		//"Tumor_019",
		//"Tumor_020",
		//"Tumor_021",
		"Tumor_022",
		"Tumor_023",
		"Tumor_024",
		"Tumor_025",
		"Tumor_026",
		"Tumor_027",
		"Tumor_028",
		"Tumor_029",
		"Tumor_030",
		"Tumor_031",
		"Tumor_032",
		"Tumor_033",
		"Tumor_034",
		"Tumor_035",
		"Tumor_036",
		"Tumor_037",
		"Tumor_038",
		"Tumor_039",
		"Tumor_040"
	};
	LBPFeatureStrategy *st = new LBPFeatureStrategy();
	for (string fileName : fileNames) {
		FeatureConstructor fc(dir, fileName, { st });
		fc.run();
	}
	delete st;
}

int main(int argc, char *argv[]) {
	//constructAll();
	//ModelTrainer m("C:/CAMELYON_TRAIN_DATA/");
	//m.trainSVM("C:/CAMELYON_TRAIN_DATA/Models/Uniform_LBP_SVM_Regression_20.yaml");
	//m.trainRF("C:/CAMELYON_TRAIN_DATA/Models/Uniform_LBP_RF_Regression_20.yaml", 50, 10);
	test("C:/CAMELYON_TRAIN_DATA/features/Uniform LBP/test", "C:/CAMELYON_TRAIN_DATA/Models/Uniform_LBP_SVM_Regression_20.yaml");
}

/*int main(int argc, char *argv[]) {
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
}*/

