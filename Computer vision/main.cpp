//
//  main.cpp
//  Computer vision
//
//  Created by Daan Grashoff on 06/02/2020.
//  Copyright © 2020 Daan Grashoff. All rights reserved.
//



#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>


using namespace std;
using namespace cv;


bool saveCameraCalibration(string fname, Mat cameraMatrix, Mat distorceCoefiientes) {
    ofstream outStream(fname);
    if (outStream) {
        uint16_t rows = cameraMatrix.rows;
        uint16_t cols = cameraMatrix.cols;
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value;
                outStream << " ";
            }
            outStream << "" << std::endl;
        }
        
        rows = distorceCoefiientes.rows;
        cols = distorceCoefiientes.cols;
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double value = distorceCoefiientes.at<double>(r, c);
                outStream << value << endl;
            }
        }
        outStream.close();
        return true;
    }
    return false;
}





//void drawAxis(cv::Mat &_image, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
//              cv::InputArray _rvec, cv::InputArray _tvec, float length) {
//    // project axis points
//    std::vector< cv::Point3f > axisPoints;
////    axisPoints.push_back(cv::Point3f(0, 0, 0));
////    axisPoints.push_back(cv::Point3f(length, 0, 0));
////    axisPoints.push_back(cv::Point3f(0, length, 0));
////    axisPoints.push_back(cv::Point3f(0, 0, length));
//    std::vector< cv::Point2f > imagePoints;
//    cv::projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);
//
//    // draw axis lines
//    cv::line(_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
//    cv::line(_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
//    cv::line(_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
//}


void drawAxes(cv::Mat &pImage, cv::InputArray pCameraMatrix, cv::InputArray pDistCoeffs,
              cv::InputArray pRvec, cv::InputArray pTvec, float pLength){
    // project axes
    std::vector< cv::Point3f > axisPoints;
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(0, pLength, 0));
    axisPoints.push_back(Point3f(pLength, pLength, 0));
    axisPoints.push_back(Point3f(pLength, 0, 0));
    axisPoints.push_back(Point3f(0, 0, -pLength));
    axisPoints.push_back(Point3f(0, pLength, -pLength));
    axisPoints.push_back(Point3f(pLength, pLength, -pLength));
    axisPoints.push_back(Point3f(pLength, 0, -pLength));
    
//    axisPoints.push_back(Point3f(0, 0, 0));
//    axisPoints.push_back(Point3f(pLength, 0, 0));
//    axisPoints.push_back(Point3f(0, pLength, 0));
//    axisPoints.push_back(Point3f(0, 0, -pLength));
    
    
//    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
//                       [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

    
    
    std::vector< cv::Point2f > imagePoints;
    cv::projectPoints(axisPoints, pRvec, pTvec, pCameraMatrix, pDistCoeffs, imagePoints);
    // draw axes
    cv::line(pImage, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(pImage, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(pImage, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    
    vector<vector<Point>> bottom;
    bottom.push_back(vector<Point>(imagePoints.begin(), imagePoints.begin()+4));
    
    drawContours(pImage, bottom, -1, Scalar(0, 0, 255), -3);
    
    for (int i = 0; i < 4; i++) {
        line(pImage, imagePoints[i], imagePoints[i + 4], Scalar(255, 255, 255), 3);
    }
    
    vector<vector<Point>> top;
    top.push_back(vector<Point>(imagePoints.begin() + 4, imagePoints.end()));
    
    drawContours(pImage, top, -1, Scalar(255, 0, 0), 3);
}




Mat draw(Mat image, vector<Point2f> corners, vector<Point2f> image_points) {
    Point2f corner = corners[0];
    
//    line(image, image_points[0], image_points[1], Scalar(255, 0, 0), 3);
//    line(image, image_points[0], image_points[2], Scalar(0, 255, 0), 3);
//    line(image, image_points[0], image_points[3], Scalar(0, 0, 255), 3);
    
    line(image, corner, image_points[1], Scalar(255, 0, 0), 3);
    line(image, corner, image_points[2], Scalar(0, 255, 0), 3);
    line(image, corner, image_points[3], Scalar(0, 0, 255), 3);
    
    return image;
}



int main() {
    
    int success = 0;
    int numOfBoards = 25;
    int numCornersHor = 10;
    int numCornersVer = 15;
    float square_size = 16.00;
    Mat img, gray, image;

    Size boardSize = Size(numCornersHor,numCornersVer);
    
    vector<vector<Point3f> > objectPoints;
    vector<vector<Point2f> > imagePoints;
    vector< Point3f > objects;
    for (int i = 0; i < numCornersVer; i++)
        for (int j = 0; j < numCornersHor; j++)
            objects.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));
    
    cv::VideoCapture cap;
    cap.open("chessboard2.mov");
    
    if( !cap.isOpened() )
    {
        std::cerr << "***Could not initialize capturing...***\n";
        std::cerr << "Current parameter's value: \n";
        return -1;
    }

    clock_t prevTimeStamp = clock();
    Mat gray_image;
    cap >> img;

    while(success < numOfBoards)
    {
        if(clock() - prevTimeStamp > 500*1e-3*CLOCKS_PER_SEC) {
            cvtColor ( img,gray, COLOR_BGR2GRAY );
            vector<Point2f> corners;
            
            bool patternfound = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
            if(patternfound)
            {
                cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.1));
                drawChessboardCorners(img, boardSize, Mat(corners), patternfound);
                imshow("win1", img);

                imagePoints.push_back(corners);
                objectPoints.push_back(objects);
                success++;
                prevTimeStamp = clock();
            }
        }
       cap >> img;
    }
    
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;

    calibrateCamera(objectPoints, imagePoints, img.size(), intrinsic, distCoeffs, rvecs, tvecs);
    

    cv::VideoCapture cap2;
    cap2.open("chessboard2.mov");
    
    Mat imageUndistorted;
    
    if( !cap2.isOpened() )
    {
        std::cerr << "***Could not initialize capturing...***\n";
        std::cerr << "Current parameter's value: \n";
        return -1;
    }

    cap2 >> image;
    Mat Rot, Tran;
    
    while(1)
    {
        undistort(image, imageUndistorted, intrinsic, distCoeffs);
        vector<Point2f> corners;
        vector< Point2f > imagePoints3;
        bool patternfound = findChessboardCorners(image, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        if(patternfound)
        {
            solvePnP(objects, corners, intrinsic, distCoeffs, Rot, Tran);
            drawAxes(imageUndistorted,intrinsic, distCoeffs, Rot, Tran,3*square_size);
        }
        
        imshow("Undistorted Image", imageUndistorted);


        cap2 >> image;
        waitKey(0);
        
    }
    
    return 0;
}
