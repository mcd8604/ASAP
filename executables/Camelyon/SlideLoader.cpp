#include <fstream>
#include "boost\filesystem.hpp"
#include "boost\algorithm\string.hpp"

#include "SlideLoader.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;

void SlideLoader::loadSlides(const string dirPath, vector<Slide> &slides, vector<string> &slideNames) {
	path dir(dirPath);
	try {
		if (exists(dir) && is_directory(dir)) {
			for (directory_entry& entry : directory_iterator(dir)) {
				path file = entry.path();
				if (is_regular_file(file)) {
					string ext = to_upper_copy(file.extension().generic_string());
					if (ext == ".YAML") {
						string imgPath = file.generic_string();
						path filePath = file.filename();
						string fileName = filePath.generic_string();
						slideNames.push_back(change_extension(fileName, "").generic_string());
						Slide s(imgPath);
						slides.push_back(s);
					}
				}
			}
		}
	}
	catch (const filesystem_error& ex) {
		cerr << ex.what() << '\n';
	}
}