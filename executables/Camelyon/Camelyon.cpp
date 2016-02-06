#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "core/filetools.h"
#include "core/CmdLineProgressMonitor.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace pathology;

int main(int argc, char *argv[]) {
  try {
    std::string inputPth;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Displays this message")
      ;
  
    po::positional_options_description positionalOptions;
    positionalOptions.add("input", 1);

    po::options_description posDesc("Positional descriptions");
    posDesc.add_options()
      ("input", po::value<std::string>(&inputPth)->required(), "Path to input")
      ;


    po::options_description descAndPos("All options");
    descAndPos.add(desc).add(posDesc);

    po::variables_map vm;
    try {
      po::store(po::command_line_parser(argc, argv).options(descAndPos)
        .positional(positionalOptions).run(),
        vm);
      if (!vm.count("input")) {
        cout << "Camelyon v1.0" << endl;
        cout << "Usage: Camelyon.exe input [options]" << endl;
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
		// Convert level 8 image to HSD color model
		// Run threshold on Density channel
		// Iterate over tiles
			// Get level 0 for tile
			// Create HSD histogram
			// Get avg, median, mode, min, max, variance, stddev, etc..
		// Output results to csv
		// Train/Test model
		// Evaluate results
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



