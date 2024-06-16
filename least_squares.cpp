#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Function to fit a quadratic polynomial surface to the background
void fitQuadraticSurface(const Mat& src, Mat& dst) {
    // Collect data points
    vector<Point3f> points;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            points.push_back(Point3f((float)x, (float)y, src.at<uchar>(y, x)));
        }
    }
    int npts = (int)points.size();

    // Fit a quadratic surface: f(x, y) = ax^2 + by^2 + cxy + dx + ey + f
    Mat A(npts, 6, CV_32F);
    Mat B(npts, 1, CV_32F);
    for (int i = 0; i < npts; i++) {
        float x = points[i].x;
        float y = points[i].y;
        float z = points[i].z;

        A.at<float>(i, 0) = x * x;
        A.at<float>(i, 1) = y * y;
        A.at<float>(i, 2) = x * y;
        A.at<float>(i, 3) = x;
        A.at<float>(i, 4) = y;
        A.at<float>(i, 5) = 1;
        B.at<float>(i, 0) = z;
    }

    Mat coeffs;
    solve(A, B, coeffs, DECOMP_SVD);

    // Create the background model
    Mat bkg = A * coeffs;
    dst = bkg.reshape(0, src.rows);
}

int main(int argc, char** argv) {
    Mat image = imread("sample_text.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    // Fit and remove the background
    Mat background, corrected, binary;
    fitQuadraticSurface(image, background);
    background.convertTo(background, CV_8U);
    subtract(background, image, corrected);
    threshold(corrected, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Display results
    imshow("Original Image", image);
    imshow("Background", background);
    imshow("Corrected Image", corrected);
    imshow("Binary Image", binary);

    waitKey(0);
    return 0;
}