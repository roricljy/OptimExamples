#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Global variables to store points and image
vector<Point2d> points;
Mat image;

// Mouse callback function to capture points
void click_event(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        points.push_back(Point2d(x, y));
        if (points.size() <= 3) {
            circle(image, Point(x, y), 5, Scalar(0, 255, 0), -1);
            imshow("Image", image);
        }
    }
}

// Function to apply the radial distortion
Point2d apply_radial_distortion(Point2d point, Point2d center, double w) {
    if (w == 0) return point;
    double ru = norm(point - center);
    double rd = atan(2 * ru * tan(w / 2)) / w;
    Point2d distorted = (point - center)* rd / ru + center;
    return distorted;
}

// Function to correct the radial distortion
Point2d correct_radial_distortion(Point2d point, Point2d center, double w) {
    if (w == 0) return point;
    double rd = norm(point - center);
    double ru = tan(rd * w) / (2 * tan(w / 2));
    Point2d corrected = (point - center) * ru / rd + center;
    return corrected;
}

// Objective function to minimize
double objective_function(const double w, const vector<Point2d>& points, Point2d center) {
    Point2d p1 = correct_radial_distortion(points[0], center, w);
    Point2d p2 = correct_radial_distortion(points[1], center, w);
    Point2d p3 = correct_radial_distortion(points[2], center, w);

    Point2d v1 = (p2 - p1) / norm(p2 - p1);
    Point2d v2 = (p3 - p1) / norm(p3 - p1);
    double loss = norm(v1.cross(v2));
    return loss;
}

// Gradient Descent implementation
double gradient_descent(const vector<Point2d>& points, Point2d center, double learning_rate = 1e-5, double tolerance = 1e-12, int max_iter = 100) {
    double w = 0;
    double eps = 1e-6;
    for (int iter = 0; iter < max_iter; iter++) {
        double w_eps = w + eps;
        double grad = (objective_function(w_eps, points, center) - objective_function(w, points, center)) / eps;
        double new_w = w - learning_rate * grad;

        double delta = fabs(new_w - w);
        if (delta < tolerance) {
            break;
        }

        w = new_w;
        double loss = objective_function(w, points, center);
        printf("[%d] loss = %lf, w = %lf\n", iter, loss, w);
    }

    return w;
}

// Main function
int main() {
    // Read the image
    image = imread("sample_radial.png");
    if (image.empty()) {
        cout << "Image not found!" << endl;
        return -1;
    }

    namedWindow("Image", WINDOW_AUTOSIZE);
    setMouseCallback("Image", click_event);

    cout << "Click on three points in the image" << endl;
    imshow("Image", image);
    waitKey(0);

    if (points.size() < 3) {
        cout << "Not enough points selected!" << endl;
        return -1;
    }

    // Perform gradient descent to find k1 and k2
    double height = image.rows;
    double width = image.cols;
    Point2d center(width / 2, height / 2);
    double w = gradient_descent(points, center);
    cout << "Found radial distortion coefficients: w = " << w << endl;

    // Apply distortion correction to the image
    Mat new_image = Mat::zeros(image.size(), image.type());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point2d distorted_point = apply_radial_distortion(Point2d(x, y), center, w);
            int xd = cvRound(distorted_point.x);
            int yd = cvRound(distorted_point.y);
            if (0 <= xd && xd < width && 0 <= yd && yd < height) {
                new_image.at<Vec3b>(y, x) = image.at<Vec3b>(yd, xd);
            }
        }
    }

    imshow("Corrected Image", new_image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}