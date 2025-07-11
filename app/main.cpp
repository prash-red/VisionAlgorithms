#ifdef USE_CUDA
#    include "algorithms/cuda_homographer.cuh"
#endif

#include <CLI/CLI.hpp>
#include <algorithms/cpu_homographer.h>
#include <constants.h>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
int run_homography(const std::string& imagePath, std::vector<int>& sourceCoords, std::vector<int>& destinationCoords,
                   const std::string& coordsFile, const std::string& outputFile, bool useCuda) {
    // If coords_file is provided, parse it to fill sourceCoords and
    // destinationCoords
    if (!coordsFile.empty()) {
        std::ifstream infile(coordsFile);
        if (!infile) {
            std::cerr << "Could not open coords file: " << coordsFile << std::endl;
            return -1;
        }
        sourceCoords.clear();
        destinationCoords.clear();
        std::string line;
        int line_count = 0;
        // Read 4 source coords (each line: x,y)
        while (line_count < 4 && std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string x_str, y_str;
            if (!std::getline(iss, x_str, ',') || !std::getline(iss, y_str)) {
                std::cerr << "Invalid source coordinate format in coords file." << std::endl;
                return -1;
            }
            sourceCoords.push_back(std::stoi(x_str));
            sourceCoords.push_back(std::stoi(y_str));
            ++line_count;
        }
        line_count = 0;
        // Read 4 destination coords (each line: x,y)
        while (line_count < 4 && std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string x_str, y_str;
            if (!std::getline(iss, x_str, ',') || !std::getline(iss, y_str)) {
                std::cerr << "Invalid destination coordinate format in coords file." << std::endl;
                return -1;
            }
            destinationCoords.push_back(std::stoi(x_str));
            destinationCoords.push_back(std::stoi(y_str));
            ++line_count;
        }
        if (sourceCoords.size() != 8 || destinationCoords.size() != 8) {
            std::cerr << "Coords file must contain 4 source and 4 destination "
                         "coordinates (x,y per line)."
                      << std::endl;
            return -1;
        }
    }

    std::array<std::array<int, Homographer::HOMOGRAPHY_2D_COORDS_SIZE>, Homographer::NUM_2D_COORDS> source{};
    std::array<std::array<int, Homographer::HOMOGRAPHY_2D_COORDS_SIZE>, Homographer::NUM_2D_COORDS> destination{};

    for (size_t i = 0; i < 4; ++i) {
        source[i][0] = sourceCoords[2 * i];
        source[i][1] = sourceCoords[2 * i + 1];
        destination[i][0] = destinationCoords[2 * i];
        destination[i][1] = destinationCoords[2 * i + 1];
    }

    std::unique_ptr<Homographer> homographer;
    if (useCuda) {
#ifdef USE_CUDA
        homographer = std::make_unique<CUDAHomographer>();
#else
        throw std::runtime_error("CUDA support is not enabled in this build");
#endif
    } else {
        homographer = std::make_unique<CPUHomographer>();
    }

    std::array<float, Homographer::HOMOGRAPHY_SIZE> homography = homographer->calculateHomography(source, destination);

    Mat image = imread(imagePath, IMREAD_COLOR);

    if (!image.data) {
        printf("No image data \n");
        return -1;
    }

    Mat outputImage;
    flip(image, image, 0);
    homographer->backwardMap(homography, image, outputImage);
    flip(outputImage, outputImage, 0);

    if (outputFile.empty()) {
        namedWindow("Display Image", WINDOW_NORMAL);
        imshow("Display Image", outputImage);
        waitKey(0);
    } else {
        imwrite(outputFile, outputImage);
    }

    return 0;
}

int setup_cli(CLI::App& app, std::string& homographyImagePath, std::vector<int>& homographySourceCoords,
              std::vector<int>& homographyDestinationCoords, std::string& homographyCoordsFilePath,
              std::string& homographyOutputImagePath, bool* useCuda) {
    // Homography subcommand
    auto homography_cmd = app.add_subcommand("homography", "Run the homography algorithm");

    homography_cmd->add_flag("--cuda", *useCuda, "Use CUDA for running the algorithms (default: false)")
        ->default_val(false)
        ->capture_default_str();

    homography_cmd->add_option("-f,--file", homographyImagePath, "The image to run the algorithm on")->required();

    auto source_option = homography_cmd
                             ->add_option("-s, --source",
                                          homographySourceCoords,
                                          "Source points for homography separated by spaces")
                             ->expected(8);

    auto destination_option = homography_cmd
                                  ->add_option("-d, --destination",
                                               homographyDestinationCoords,
                                               "Destination points for homography separated by spaces")
                                  ->expected(8);

    homography_cmd
        ->add_option("-c, --coords-file",
                     homographyCoordsFilePath,
                     "File containing comma seperated source and destination coordinates on each new line")
        ->excludes(source_option)
        ->excludes(destination_option);

    homography_cmd->add_option("-o, --output-file",
                               homographyOutputImagePath,
                               "The file to save the transformed image to. If this option is not defined, "
                               "the transformed image will be displayed in a new window");

    return 0;
}

int main(int argc, char** argv) {
    CLI::App app{"A CLI tool for computer vision algorithms"};

    std::string homographyImagePath;
    std::vector<int> homographySourceCoords = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS;
    std::vector<int> homographyDestinationCoords = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS;
    std::string homographyCoordsFilePath;
    std::string homographyOutputImagePath;
    bool useCuda = false;

    setup_cli(app,
              homographyImagePath,
              homographySourceCoords,
              homographyDestinationCoords,
              homographyCoordsFilePath,
              homographyOutputImagePath,
              &useCuda);

    CLI11_PARSE(app, argc, argv);

    auto homography_cmd = app.get_subcommand("homography");
    if (homography_cmd && *homography_cmd) {
        return run_homography(homographyImagePath,
                              homographySourceCoords,
                              homographyDestinationCoords,
                              homographyCoordsFilePath,
                              homographyOutputImagePath,
                              useCuda);
    }

    // TODO: add other algorithms here

    std::cerr << "No valid subcommand provided." << std::endl;
    return 1;
}
