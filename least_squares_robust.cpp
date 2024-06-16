#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Cauchy weight function
double CauchyWeight(double residual, double c) {
    return 1.0 / (1.0 + (residual / c) * (residual / c));
}

// Function to fit a quadratic polynomial surface to the background using Cauchy weights
void fitQuadraticSurfaceRobust(const Mat& src, Mat& dst, double c) {
    // Collect data points
    vector<Point3f> points;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            points.push_back(Point3f((float)x, (float)y, src.at<uchar>(y, x)));
        }
    }
    int npts = (int)points.size();

    // Initialize matrices
    Mat A(npts, 6, CV_32F);
    Mat B(npts, 1, CV_32F);
    for (int i = 0; i < npts; ++i) {
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

    // Initial guess for the coefficients (least squares)
    Mat coeffs;
    solve(A, B, coeffs, DECOMP_SVD);

    // Iterative reweighted least squares
    Mat W = Mat::eye(npts, npts, CV_32F);
    for (int iter = 0; iter < 10; ++iter) {  // 10 iterations for example
        // Calculate residuals
        Mat residuals = A * coeffs - B;

        // Calculate weights
        for (int i = 0; i < npts; i++) {
            W.at<float>(i, i) = (float)CauchyWeight(residuals.at<float>(i, 0), c);
        }

        // Weighted least squares
        Mat Aw = A.t() * W * A;
        Mat Bw = A.t() * W * B;
        solve(Aw, Bw, coeffs, DECOMP_SVD);
    }

    // Create the background model
    Mat bkg = A * coeffs;
    dst = bkg.reshape(0, src.rows);
}

int main(int argc, char** argv) {
    // Load the input image
    Mat image = imread("sample_circle.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    // Fit and remove the background
    Mat background, imagef, residual, binary;
    double c = 2.3849; // Cauchy function parameter
    fitQuadraticSurfaceRobust(image, background, c);
    image.convertTo(imagef, CV_32F);
    subtract(background, imagef, residual);
    background.convertTo(background, CV_8U);
    residual.convertTo(residual, CV_8U);
    threshold(residual, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Display results
    imshow("Original Image", image);
    imshow("Background", background);
    imshow("Residual", residual);
    imshow("Binary Image", binary);

    waitKey(0);
    return 0;
}