//
// Created by miguelyermo on 1/3/20.
//

/*
* FILENAME :  handlers.h  
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*  
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 03:07 1/3/20
*
*/

#ifndef CPP_HANDLERS_H
#define CPP_HANDLERS_H

#include "readers/FileReaderFactory.hpp"
#include <filesystem>
#include <fstream>
#include <lasreader.hpp>
#include "Geometry/point.hpp"
#include "Geometry/PointMetadata.hpp"
#include <optional>
#include "TimeWatcher.hpp"
#include "point_containers.hpp"

namespace fs = std::filesystem;

/**
 * This function creates a directory if it does not exist.
 * @param dirname
 * @return
 */
void createDirectory(const fs::path& dirName) {
	if (!fs::is_directory(dirName)) { fs::create_directories(dirName); }
}

template<PointContainer Container>
void pointCloudReadLog(const Container &points, TimeWatcher &tw, const fs::path& fileName) {
    auto mem_size = (sizeof(Container) + (sizeof(Point) * points.size())) / (1024.0 * 1024.0);
    const std::string mem_size_str = std::to_string(mem_size) + " MB";
    const std::string point_size_str =  std::to_string(sizeof(Point)) + " bytes";
    const std::string time_elapsed_str = std::to_string(tw.getElapsedDecimalSeconds()) + " seconds";
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Point cloud read:"           << std::setw(LOG_FIELD_WIDTH) << fileName.stem()                   			  << "\n";
    std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Time to read:"               << std::setw(LOG_FIELD_WIDTH) << time_elapsed_str                               << "\n";
    std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of read points:"      << std::setw(LOG_FIELD_WIDTH) << points.size()                                  << "\n";
    std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Size of point type:"         << std::setw(LOG_FIELD_WIDTH) << point_size_str                                 << "\n";
    std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Points vector size:"         << std::setw(LOG_FIELD_WIDTH) << mem_size_str                                   << "\n";
    // TODO std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Alligned to cache lines?:"  	<< std::setw(LOG_FIELD_WIDTH) << (checkMemoryAlligned(points) ? "Yes" : "No") << "\n";
    std::cout << std::endl;
}

/// @brief The optional will be null if Point_t does not require metadata (is not Point)
template <PointContainer Container>
std::pair<Container, std::optional<std::vector<PointMetadata>>> readPoints(const fs::path& fileName) {
	// Open the file and create the reader
	auto fExt = fileName.extension();
	FileReader_t readerType = chooseReaderType(fExt);
	if (readerType == err_t)
	{
		std::cout << "Uncompatible file format\n";
		exit(-1);
	}
	std::shared_ptr<FileReader<Container>> fileReader = FileReaderFactory::makeReader<Container>(readerType, fileName);
	TimeWatcher tw;
    tw.start();
    auto [points, metadata] = fileReader->readMeta();
    tw.stop();
    pointCloudReadLog(points, tw, fileName);
    return std::make_pair(points, std::optional<std::vector<PointMetadata>>(metadata));
}


#endif //CPP_HANDLERS_H
