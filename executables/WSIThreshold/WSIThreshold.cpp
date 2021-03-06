#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "ThresholdWholeSlideFilter.h"
#include "core/filetools.h"
#include "core/CmdLineProgressMonitor.h"
#include "config/pathology_config.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace pathology;

int main(int argc, char *argv[]) {
  try {
    std::string inputPth, outputPth;
    unsigned int processedLevel;
    int component;
    float lowerThreshold, upperThreshold;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Displays this message")
      ("level,l", po::value<unsigned int>(&processedLevel)->default_value(0), "Set the level to be processed")
      ("component,c", po::value<int>(&component)->default_value(-1), "Color component to select for threshold, if none, threshold all.")
      ("lower_threshold,ll", po::value<float>(&lowerThreshold)->default_value(std::numeric_limits<float>::min()), "Set the lower threshold")
      ("upper_threshold,ul", po::value<float>(&upperThreshold)->default_value(std::numeric_limits<float>::max()), "Set the upper threshold")
      ;
  
    po::positional_options_description positionalOptions;
    positionalOptions.add("input", 1);
    positionalOptions.add("output", 1);

    po::options_description posDesc("Positional descriptions");
    posDesc.add_options()
      ("input", po::value<std::string>(&inputPth)->required(), "Path to input")
      ("output", po::value<std::string>(&outputPth)->default_value("."), "Path to output")
      ;


    po::options_description descAndPos("All options");
    descAndPos.add(desc).add(posDesc);

    po::variables_map vm;
    try {
      po::store(po::command_line_parser(argc, argv).options(descAndPos)
        .positional(positionalOptions).run(),
        vm);
      if (!vm.count("input")) {
        cout << "WSIThreshold v" << ASAP_VERSION_STRING << endl;
        cout << "Usage: WSIThreshold.exe input output [options]" << endl;
      }
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
      }
      po::notify(vm);
    }
    catch (boost::program_options::required_option& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << "Use -h or --help for usage information" << std::endl;
      return 1;
    }
    MultiResolutionImageReader reader; 
    MultiResolutionImage* input = reader.open(inputPth);
    CmdLineProgressMonitor monitor;
    if (input) {
      ThresholdWholeSlideFilter fltr;
      fltr.setInput(input);
      fltr.setOutput(outputPth);
      fltr.setProgressMonitor(&monitor);
      fltr.setLowerThreshold(lowerThreshold);
      fltr.setUpperThreshold(upperThreshold);
      fltr.setProcessedLevel(processedLevel);
      if (!fltr.process()) {
        std::cerr << "ERROR: Processing failed" << std::endl;
      }
      delete input;
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
  } 
  catch (std::exception& e) {
    std::cerr << "Unhandled exception: "
      << e.what() << ", application will now exit" << std::endl;
    return 2;
  }
	return 0;
}



