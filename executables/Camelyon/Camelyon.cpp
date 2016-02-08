#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "core/filetools.h"
#include "core/CmdLineProgressMonitor.h"
#include "config/pathology_config.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace pathology;
using namespace cv;

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
		// TODO Convert level 8 image to HSD color model
		// TODO Run threshold on Density channel

		// Iterate over tiles

		// Assume 512 always fits integrally
		int tileWidth = 512;
		int tileHeight = 512;
		vector<unsigned long long, allocator<unsigned long long>> dim = input->getLevelDimensions(0);
		int numTilesX = dim[0] / tileWidth;
		int numTilesY = dim[1] / tileHeight;

		for (int y = 0; y < numTilesY; y++) {
			for (int x = 0; x < numTilesX; x++) {
				// Get level 0 for tile
				Patch<double> p = input->getPatch<double>(x * tileWidth, y * tileHeight, tileWidth, tileHeight, 0);
				Mat m = patchToMat(p);
				m.convertTo(m, CV_32F);

				// TODO Convert to HSD
				// Create histogram 
				Mat rgbHist;
				int histSize = 256;
				int channels[] = { 0, 1, 2 };
				float range[] = { 0, 256 };
				const float* ranges[] = { range };
				calcHist(&m, 1, channels, Mat(), rgbHist, 1, &histSize, ranges, true, false);
				//cout << "Histo Created for (" << x << "," << y << ")\n";

				// TODO Get avg, median, mode, min, max, variance, stddev, etc..
			}
		}
		// TODO Output results to csv
		// TODO Train/Test model
		// TODO Evaluate results
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



