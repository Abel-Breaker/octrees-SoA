#pragma once
#include <omp.h>
#include <type_traits>
#include "benchmarking.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "TimeWatcher.hpp"
#include "main_options.hpp"
#include "unibnOctree.hpp"
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "pcl_cloud_reader.hpp"
#endif
#include "nanoflann.hpp"
#include "nanoflann_wrappers.hpp"
#include "papi.h"
#include "encoding_log.hpp"
#include "build_log.hpp"
#include "../PointEncoding/point_encoder_factory.hpp"
#include "../PointEncoding/point_encoder.hpp"
#include "point_containers.hpp"

template <PointContainer Container>
class EncodingBuildBenchmarks {
    using key_t = size_t;
    private:
        Container& points;
        std::optional<std::vector<PointMetadata>> &metadata;
        std::vector<key_t> codes;
        std::ostream& outputEncoding;
        std::ostream& outputBuild;
        Box box;

    public:
        EncodingBuildBenchmarks(Container &points, std::optional<std::vector<PointMetadata>> &metadata, 
            std::ostream &outputEncoding, std::ostream& outputBuild): 
            points(points), metadata(metadata), outputEncoding(outputEncoding), outputBuild(outputBuild) {}
        
        void runBuildBenchmark(SearchStructure structure, EncoderType encoding) {
            int eventSet = PAPI_NULL;
            auto [events, descriptions] = buildCombinedEventList();
            std::vector<long long> eventValues(events.size());
            auto &enc = PointEncoding::getEncoder(encoding);
            std::shared_ptr<BuildLog> log = std::make_shared<BuildLog>();
            log->encoding = encoding;
            log->cloudSize = points.size();
            log->maxLeafPoints = mainOptions.maxPointsLeaf;
            log->structure = structure;

            // Initialize PAPI
            if(mainOptions.cacheProfiling) {
                auto [events, descriptions] = buildCombinedEventList();
                eventSet = initPapiEventSet(events);
                if (eventSet == PAPI_NULL) {
                    std::cout << "Failed to initialize PAPI event set." << std::endl;
                    exit(1);
                }
            }

            // Run the build for the chosen structure
            switch(structure) {
                case SearchStructure::PTR_OCTREE: {
                    size_t currRepeat = 0;
                    auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                        Octree oct(points, box);
                    }, mainOptions.useWarmup, eventSet, eventValues.data());
                    log->buildTime = stats.mean();
                    // extra build for logging (not counted towards total time)
                    Octree oct(points, box);
                    oct.logOctreeData(log);
                    break;
                }
                case SearchStructure::LINEAR_OCTREE: {
                    double totalLeaf = 0.0, totalInternal = 0.0;
                    if(enc.getShortEncoderName() == encoderTypeToString(EncoderType::NO_ENCODING)) {
                        std::cout << "  Skipping Linear Octree since point cloud was not reordered!" << std::endl;
                        return;
                    } else {
                        bool insideWarmup = mainOptions.useWarmup;
                        auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                            LinearOctree oct(points, codes, box, enc, log);
                            if(insideWarmup)
                                totalLeaf += log->linearOctreeLeafTime, totalInternal += log->linearOctreeInternalTime;
                            insideWarmup = false;
                        }, mainOptions.useWarmup, eventSet, eventValues.data());
                        log->linearOctreeLeafTime = totalLeaf / mainOptions.repeats;
                        log->linearOctreeInternalTime = totalInternal / mainOptions.repeats;
                        log->buildTime = log->linearOctreeLeafTime + log->linearOctreeInternalTime;
                    }
                    break;
                }
#ifdef HAVE_PCL
                case SearchStructure::PCL_KDTREE: {
                    auto pclCloud = convertCloudToPCL(points);
                    // Build the PCL Kd-tree
                    auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree = pcl::KdTreeFLANN<pcl::PointXYZ>();
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                        kdtree.setInputCloud(cloudPtr);
                    }, mainOptions.useWarmup, eventSet, eventValues.data());
                    log->buildTime = stats.mean();
                    break;
                }
                case SearchStructure::PCL_OCTREE: {
                    auto pclCloud = convertCloudToPCL(points);
                    auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                            // Build the PCL Octree
                            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oct(mainOptions.pclOctResolution);
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                            oct.setInputCloud(cloudPtr);
                            oct.addPointsFromInputCloud();
                    }, mainOptions.useWarmup, eventSet, eventValues.data());
                    log->buildTime = stats.mean();
                    break;
                }
#endif
                case SearchStructure::UNIBN_OCTREE: {
                    auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                        unibn::Octree<Point, Container> oct;
                        unibn::OctreeParams params;
                        params.bucketSize = mainOptions.maxPointsLeaf;
                        oct.initialize(points, params);
                    }, mainOptions.useWarmup, eventSet, eventValues.data());
                    log->buildTime = stats.mean();
                    break;
                }
                case SearchStructure::NANOFLANN_KDTREE: {
                    NanoflannPointCloud<Container> npc(points);
                    auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
                        NanoFlannKDTree<Container> kdtree(3, npc, {mainOptions.maxPointsLeaf});
                    }, mainOptions.useWarmup, eventSet, eventValues.data());
                    log->buildTime = stats.mean();
                    break;
                }
            }
            if(mainOptions.cacheProfiling) {
                log->l1dMisses = eventValues[0];
                log->l2dMisses = eventValues[1];
                log->l3Misses = eventValues[2];
            }
            log->toCSV(outputBuild);
        }

        void runEncodingBenchmark(EncoderType encoding) {
            double totalBbox = 0.0, totalEnc = 0.0, totalSort = 0.0;
            std::shared_ptr<EncodingLog> log = std::make_shared<EncodingLog>();
            auto& enc = PointEncoding::getEncoder(encoding);
            if(mainOptions.useWarmup) {
                auto pointsCopy = points;
                auto [codesWarmup, boxWarmup] = enc.sortPoints(pointsCopy, metadata, log);
            }
            for(int i = 0; i<mainOptions.repeats; i++) {
                auto pointsCopy = points;
                auto [codesRepeat, boxRepeat] = enc.sortPoints(pointsCopy, metadata, log);
                totalBbox += log->boundingBoxTime;
                totalEnc += log->encodingTime;
                totalSort += log->sortingTime;
                if(i == mainOptions.repeats - 1) {
                    points = pointsCopy;
                    codes = codesRepeat;
                    box = boxRepeat;
                }
            }
            log->cloudSize = points.size();
            log->encoding = encoding;
            log->boundingBoxTime = totalBbox / mainOptions.repeats;
            log->encodingTime = totalEnc / mainOptions.repeats;
            log->sortingTime = totalSort / mainOptions.repeats;
            log->toCSV(outputEncoding);
        }

        /// @brief Main benchmarking function
        void runEncodingBuildBenchmarks() {
            BuildLog::writeCSVHeader(outputBuild);
            EncodingLog::writeCSVHeader(outputEncoding);
            int currentEncoder = 1;
            int totalStructureBenchmarks = mainOptions.searchStructures.size();

            // First, we do unencoded points, since we don't reorder here and later the points are altered. 
            // If NO_ENCODING was done after Hilbert per example, the array would still be on Hilbert order.
            if(mainOptions.encodings.contains(EncoderType::NO_ENCODING)) {
                std::cout << "Running encoding benchmark on " << encoderTypeToString(EncoderType::NO_ENCODING) 
                    << " (" << currentEncoder++ << " out of " << mainOptions.encodings.size() << ")\n";
                runEncodingBenchmark(EncoderType::NO_ENCODING);
                int currentStructure = 1;
                for(SearchStructure structure: mainOptions.searchStructures) {
                    std::cout << "  Running structure benchmark on " << searchStructureToString(structure) 
                        << " (" << currentStructure++ << " out of " << mainOptions.searchStructures.size() << ")\n";
                        runBuildBenchmark(structure, EncoderType::NO_ENCODING);
                }
            }

            for(EncoderType encoding: mainOptions.encodings) {
                // Skip NO_ENCODING, already done
                if(encoding == EncoderType::NO_ENCODING)
                    continue;
                std::cout << "Running encoding benchmark on " << encoderTypeToString(encoding) 
                    << " (" << currentEncoder++ << " out of " << mainOptions.encodings.size() << ")\n";
                runEncodingBenchmark(encoding);
                int currentStructure = 1;
                for(SearchStructure structure: mainOptions.searchStructures) {
                    std::cout << "  Running structure benchmark on " << searchStructureToString(structure) 
                        << " (" << currentStructure++ << " out of " << mainOptions.searchStructures.size() << ")\n";
                        runBuildBenchmark(structure, encoding);
                }
            }
        }
};
