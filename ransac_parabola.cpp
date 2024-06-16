#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

struct ParabolaModel {
    float a, b, c;  // Coefficients of the parabola y = ax^2 + bx + c
};

// Function to fit a parabola to given points using least squares
ParabolaModel fitParabola(const vector<Point2f>& points) {
    int n = points.size();
    Mat A(n, 3, CV_32F);
    Mat B(n, 1, CV_32F);
    for (int i = 0; i < n; ++i) {
        float x = points[i].x;
        float y = points[i].y;
        A.at<float>(i, 0) = x * x;
        A.at<float>(i, 1) = x;
        A.at<float>(i, 2) = 1;
        B.at<float>(i, 0) = y;
    }
    Mat coeffs;
    solve(A, B, coeffs, DECOMP_SVD);
    ParabolaModel model = { coeffs.at<float>(0, 0), coeffs.at<float>(1, 0), coeffs.at<float>(2, 0) };
    return model;
}

// Function to compute the distance of a point from the parabola
float pointToParabolaDistance(const Point2f& point, const ParabolaModel& model) {
    float x = point.x;
    float y = point.y;
    float y_fit = model.a * x * x + model.b * x + model.c;
    return fabs(y - y_fit);
}

// RANSAC algorithm to fit a parabola to 2D points
ParabolaModel ransacParabola(const vector<Point2f>& points, int maxIter, float threshold, int minInliers) {
    int bestInlierCount = 0;
    ParabolaModel bestModel;

    srand(static_cast<unsigned int>(time(0)));  // Seed the random number generator

    for (int iter = 0; iter < maxIter; ++iter) {
        // Randomly select 3 points
        vector<Point2f> sample;
        for (int i = 0; i < 3; ++i) {
            sample.push_back(points[rand() % points.size()]);
        }

        // Fit a parabola to the sample
        ParabolaModel model = fitParabola(sample);

        // Count inliers
        int inlierCount = 0;
        for (const auto& point : points) {
            if (pointToParabolaDistance(point, model) < threshold) {
                ++inlierCount;
            }
        }

        // Update the best model if the current model has more inliers
        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestModel = model;
        }
    }

    // Refine the model using all inliers
    vector<Point2f> inliers;
    for (const auto& point : points) {
        if (pointToParabolaDistance(point, bestModel) < threshold) {
            inliers.push_back(point);
        }
    }
    if (inliers.size() >= minInliers) {
        bestModel = fitParabola(inliers);
    }

    return bestModel;
}

int main() {
    // Generate some sample data with outliers
    vector<Point2f> points = {
        {0, 1}, {1, 2}, {2, 5}, {3, 10}, {4, 17}, {5, 26},
        {1, 8}, {2, -1}, {3, 4}, {5, 12}, {6, 30}, {7, 5}
    };

    // RANSAC parameters
    int maxIter = 1000;
    float threshold = 1.0;
    int minInliers = 5;

    // Fit the parabola using RANSAC
    ParabolaModel model = ransacParabola(points, maxIter, threshold, minInliers);

    // Print the model coefficients
    cout << "Fitted parabola: y = " << model.a << "x^2 + " << model.b << "x + " << model.c << endl;

    // Visualize the points and the fitted parabola
    Mat plot(400, 600, CV_8UC3, Scalar(255, 255, 255));
    for (const auto& point : points) {
        circle(plot, Point(point.x * 50 + 50, 400 - point.y * 10 - 50), 5, Scalar(0, 0, 255), -1);
    }
    for (float x = 0; x < 10; x+=0.1) {
        float y = model.a * x * x + model.b * x + model.c;
        circle(plot, Point(x * 50 + 50, 400 - y * 10 - 50), 3, Scalar(255, 0, 0), -1);
    }

    imshow("RANSAC Parabola Fitting", plot);
    waitKey(0);

    return 0;
}