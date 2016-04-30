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
#include "boost\program_options.hpp"

using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	bool train;
	bool test;
	string rfModelFile;
	int rfMaxDepth;
	int rfNumTrees;
	string svmModelFile;
	string trainDataDirectory;
	string testDataDirectory;

	options_description optionsDesc("Options");
	optionsDesc.add_options()
		("help,h", "Displays this message")
		("train,n", value<bool>(&train)->default_value(true), "Set to True to train new models.")
		("test,t", value<bool>(&test)->default_value(true), "Set to True to test models.")
		("rfModelFile,r", value<string>(&rfModelFile)->default_value("rf.yaml"), "Set the file path for the model.")
		("rfMaxDepth,rd", value<int>(&rfMaxDepth)->default_value(10), "Set the max depth of the Random Forest model.")
		("rfNumTrees,rt", value<int>(&rfNumTrees)->default_value(50), "Set the number of trees for the Random Forest model.")
		("svmModelFile,s", value<string>(&svmModelFile)->default_value("svm.yaml"), "Set the file path for the model.")
		("trainDataDirectory,nd", value<string>(&trainDataDirectory)->default_value("trainData"), "The directory containing training data.")
		("testDataDirectory,td", value<string>(&testDataDirectory)->default_value("testData"), "The directory containing test data.")
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
	
	if (train) {
		ModelTrainer m(trainDataDirectory);
		m.trainSVM(svmModelFile);
		m.trainRF(rfModelFile, rfNumTrees, rfMaxDepth);
	} 
   
	if (test) {
		ModelTester tester;
		tester.loadSVMModel(svmModelFile);
		tester.Test(testDataDirectory, testDataDirectory + "/SVM_RESULTS/");
		tester.loadRFModel(rfModelFile);
		tester.Test(testDataDirectory, testDataDirectory + "/RF_RESULTS/");
	}

	return 0;
}

