#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImageWriter.h"
#include "TIFFImage.h"
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

/*template <typename T>
double getAvgDensity<T>(Mat *img, Rect r) {
	double s = 0;
	for (int y = 0; y < r.height; y++)
		for (int x = 0; x < r.width; x++)
			s += img->at<T>(Point(r.x + x, r.y + y));
	return s / r.area;
}*/

bool isForeground(Mat *img, Rect r, double densityThresh = 0.2, double backgroundThresh = 0.1) {
	double count = 0.0;
	for (int y = 0; y < r.height; y++)
		for (int x = 0; x < r.width; x++)
			if (img->at<double>(Point(r.x + x, r.y + y)) > densityThresh)
				count++;
	return (count / (r.width * r.height)) > backgroundThresh;
}

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
		// Assume input has a lvl 8
		int lowResLvl = 8;
		vector<unsigned long long, allocator<unsigned long long>> lowResDim = input->getLevelDimensions(8);
		Patch<uchar> lowResPatch = input->getPatch<uchar>(0, 0, lowResDim[0], lowResDim[1], lowResLvl);
		Mat lowResMat = patchToMat(lowResPatch);
		cvtColor(lowResMat, lowResMat, COLOR_BGR2HSV);
		imshow("HSV", lowResMat);

		// Convert level 8 image to HSD color model
		ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
		Patch<double> cxPatch, cyPatch, dPatch;
		filter->filter(lowResPatch, cxPatch);
		filter->setOutputStain(1);
		filter->filter(lowResPatch, cyPatch);
		filter->setOutputStain(2);
		filter->filter(lowResPatch, dPatch);
		Mat cX = patchToMat(cxPatch);
		Mat cY = patchToMat(cyPatch);
		Mat d = patchToMat(dPatch);
		
		// (temporary check)
		imshow("cX", cX);
		imshow("cY", cY);
		imshow("D", d);
		waitKey(0);
		
		// Assume 512 always fits integrally
		int tileWidth = 512;
		int tileHeight = 512;
		int lowResTileWidth = tileWidth / pow(2, lowResLvl);
		int lowResTileHeight = tileHeight / pow(2, lowResLvl);
		vector<unsigned long long, allocator<unsigned long long>> dim = input->getLevelDimensions(0);
		int numTilesX = dim[0] / tileWidth;
		int numTilesY = dim[1] / tileHeight;

		// Iterate over foreground tiles
		for (int y = 0; y < numTilesY; y++) {
			for (int x = 0; x < numTilesX; x++) {
				Rect r(x * lowResTileWidth, y * lowResTileHeight, lowResTileWidth, lowResTileHeight);
				if (isForeground(&d, r, 0.2, 0.1)) {
					// Get level 0 for tile
					Patch<uchar> p = input->getPatch<uchar>(x * tileWidth, y * tileHeight, tileWidth, tileHeight, 0);
					Mat m = patchToMat(p);

					// TODO Convert to HSD
					// Create histogram 
					Mat rgbHist;
					int histSize = 256;
					int channels[] = { 0, 1, 2 };
					float range[] = { 0, 256 };
					const float* ranges[] = { range };
					calcHist(&m, 1, channels, Mat(), rgbHist, 1, &histSize, ranges, true, false);
					cout << "Histo Created for (" << x << "," << y << ")\n";

					// TODO Get avg, median, mode, min, max, variance, stddev, etc..
				}
			}
		}

		delete filter;

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

