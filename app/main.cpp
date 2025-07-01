// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef ENABLE_DOCTEST_IN_LIBRARY
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <homography.h>

#include <CLI/CLI.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
int main (int argc, char** argv) {
    CLI::App app{ "A CLI tool for computer vision algorithms" };

    std::string file_path;
    std::vector sourceCoords      = { 0, 0, 1, 0, 1, 1, 0, 1 };
    std::vector destinationCoords = { 0, 0, 2, 0, 2, 2, 0, 2 };
    std::string coords_file;
    std::string output_file;

    auto homography_cmd = app.add_subcommand ("homography", "Run the homography algorithm");
    homography_cmd
    ->add_option ("-f,--file", file_path, "The image to run the algorithm on")
    ->required ();

    auto source_option = homography_cmd
                         ->add_option ("-s, --source", sourceCoords,
                         "Source points for homography separated by spaces")
                         ->expected (8);

    auto destination_option =
    homography_cmd
    ->add_option ("-d, --destination", destinationCoords,
    "Destination points for homography separated by spaces")
    ->expected (8);

    homography_cmd
    ->add_option ("-c, --coords-file",
    coords_file, "File containing comma seperated source and destination coordinates on each new line")
    ->excludes (source_option)
    ->excludes (destination_option);

    homography_cmd->add_option ("-o, --output-file", output_file,
    "The file to save the transformed image to. If this option is not defined, "
    "the transformed image will be displayed in a new window");

    CLI11_PARSE (app, argc, argv);

    // If coords_file is provided, parse it to fill sourceCoords and destinationCoords
    if (!coords_file.empty ()) {
        std::ifstream infile (coords_file);
        if (!infile) {
            std::cerr << "Could not open coords file: " << coords_file << std::endl;
            return -1;
        }
        sourceCoords.clear ();
        destinationCoords.clear ();
        std::string line;
        int line_count = 0;
        // Read 4 source coords (each line: x,y)
        while (line_count < 4 && std::getline (infile, line)) {
            std::istringstream iss (line);
            std::string x_str, y_str;
            if (!std::getline (iss, x_str, ',') || !std::getline (iss, y_str)) {
                std::cerr << "Invalid source coordinate format in coords file."
                          << std::endl;
                return -1;
            }
            sourceCoords.push_back (std::stoi (x_str));
            sourceCoords.push_back (std::stoi (y_str));
            ++line_count;
        }
        line_count = 0;
        // Read 4 destination coords (each line: x,y)
        while (line_count < 4 && std::getline (infile, line)) {
            std::istringstream iss (line);
            std::string x_str, y_str;
            if (!std::getline (iss, x_str, ',') || !std::getline (iss, y_str)) {
                std::cerr
                << "Invalid destination coordinate format in coords file." << std::endl;
                return -1;
            }
            destinationCoords.push_back (std::stoi (x_str));
            destinationCoords.push_back (std::stoi (y_str));
            ++line_count;
        }
        if (sourceCoords.size () != 8 || destinationCoords.size () != 8) {
            std::cerr << "Coords file must contain 4 source and 4 destination "
                         "coordinates (x,y per line)."
                      << std::endl;
            return -1;
        }
    }

    std::array<std::array<int, Homographer::HOMOGRAPHY_2D_COORDS_SIZE>, Homographer::NUM_2D_COORDS> source{};
    std::array<std::array<int, Homographer::HOMOGRAPHY_2D_COORDS_SIZE>, Homographer::NUM_2D_COORDS> destination{};

    for (size_t i = 0; i < 4; ++i) {
        source[i][0]      = sourceCoords[2 * i];
        source[i][1]      = sourceCoords[2 * i + 1];
        destination[i][0] = destinationCoords[2 * i];
        destination[i][1] = destinationCoords[2 * i + 1];
    }

    array<float, Homographer::HOMOGRAPHY_SIZE> homography =
    Homographer::calculateHomography (source, destination);

    Mat image = imread (file_path, IMREAD_COLOR);

    if (!image.data) {
        printf ("No image data \n");
        return -1;
    }

    Mat outputImage;
    flip (image, image, 0);
    Homographer::backwardMap (homography, image, outputImage);
    flip (outputImage, outputImage, 0);

    if (output_file.empty ()) {
        namedWindow ("Display Image", WINDOW_NORMAL);
        imshow ("Display Image", outputImage);
        waitKey (0);
    } else {
        imwrite (output_file, outputImage);
    }

    return 0;
}
