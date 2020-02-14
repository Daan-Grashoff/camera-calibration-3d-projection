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


void drawAxes(cv::Mat &pImage, cv::InputArray pCameraMatrix, cv::InputArray pDistCoeffs,
              cv::InputArray pRvec, cv::InputArray pTvec, float pLength){
    // cubePoints
    vector<Point3f> cubePoints;
    cubePoints.push_back(Point3f(0, 0, 0));
    cubePoints.push_back(Point3f(0, pLength, 0));
    cubePoints.push_back(Point3f(pLength, pLength, 0));
    cubePoints.push_back(Point3f(pLength, 0, 0));
    cubePoints.push_back(Point3f(0, 0, -pLength));
    cubePoints.push_back(Point3f(0, pLength, -pLength));
    cubePoints.push_back(Point3f(pLength, pLength, -pLength));
    cubePoints.push_back(Point3f(pLength, 0, -pLength));
    //    Axis points;
    vector<Point3f> axisPoints;
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(pLength*2, 0, 0));
    axisPoints.push_back(Point3f(0, pLength*2, 0));
    axisPoints.push_back(Point3f(0, 0, -pLength*2));
    
    
    
    //    project cube;
    vector<Point2f> imagePoints;
    projectPoints(cubePoints, pRvec, pTvec, pCameraMatrix, pDistCoeffs, imagePoints);
    //    project axis;
    vector<Point2f> imagePoints2;
    projectPoints(axisPoints, pRvec, pTvec, pCameraMatrix, pDistCoeffs, imagePoints2);
    
    // draw axes;
    line(pImage, imagePoints2[0], imagePoints2[1], Scalar(0, 0, 255), 3);
    line(pImage, imagePoints2[0], imagePoints2[2], Scalar(0, 255, 0), 3);
    line(pImage, imagePoints2[0], imagePoints2[3], Scalar(255, 0, 0), 3);
    
    //    Draw bottom of cube;
    vector<vector<Point>> bottom;
    bottom.push_back(vector<Point>(imagePoints.begin(), imagePoints.begin()+4));
    drawContours(pImage, bottom, -1, Scalar(0, 0, 255), -3);
    
    //    Draw pilars of cube;
    for (int i = 0; i < 4; i++) {
        line(pImage, imagePoints[i], imagePoints[i + 4], Scalar(255, 255, 255), 3);
    }
    
    //    Draw top of cube;
    vector<vector<Point>> top;
    top.push_back(vector<Point>(imagePoints.begin() + 4, imagePoints.end()));
    drawContours(pImage, top, -1, Scalar(255, 0, 0), 3);
}

int main() {
    int success = 0;
    int numOfBoards = 30;
    int numCornersHor = 10;
    int numCornersVer = 15;
    float square_size = 16.00;
    Mat image, grayImage, imageChessCorners;
    
    Size boardSize = Size(numCornersHor,numCornersVer);
    
    vector<vector<Point3f> > objectPoints;
    vector<vector<Point2f> > imagePoints;
    vector< Point3f > objects;
    
    //    Create world objects with the sizes of the squares.
    for (int i = 0; i < numCornersVer; i++)
        for (int j = 0; j < numCornersHor; j++)
            objects.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));
    
    //    Using a video to calibrate the camera.;
    //    Stop when there is no video;
    VideoCapture cap = VideoCapture(0);
    //    cap.open(0);
    if( !cap.isOpened() )
    {
        cerr << "***Could not initialize capturing...***\n";
        cerr << "Current parameter's value: \n";
        return -1;
    }
    
    //  Creating a clock so the snapshots of the chessboard are not alike.;
    clock_t prevTimeStamp = clock();
    
    
    cap >> image;
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    
    //    Finding patterns;
    //    While the amount of found chessboards is lower than the prefered amount of chessboards;
    while(success < numOfBoards)
    {
        imageChessCorners = image;
        //            using a grayscale image, so the calculations are less intensive.;
        cvtColor ( image,grayImage, COLOR_BGR2GRAY );
        vector<Point2f> corners;
        //            Finding chessboard patterns;
        bool patternfound = findChessboardCorners(grayImage, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        if(patternfound)
        {
            //                improving corners;
            cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1),TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.1));
            //                Draw the chessboard corners;
            drawChessboardCorners(imageChessCorners, boardSize, Mat(corners), patternfound);
            
            
            //        Saving the corners and world objects for calibration;
            //        using the clock to make steps of .5 seconds;
            
            if(clock() - prevTimeStamp > 500*1e-3*CLOCKS_PER_SEC) {
                imwrite("./"+to_string(numOfBoards)+"-2/image"+to_string(success)+".jpg", image);
                imagePoints.push_back(corners);
                objectPoints.push_back(objects);
                success++;
                prevTimeStamp = clock();
            }
        }
        imshow("win1", imageChessCorners);
        imshow("win1", image);
        cap >> image;
        
        
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    
    
    //    Calibration;
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    //    Calibrate camera;
    calibrateCamera(objectPoints, imagePoints, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
    
    
    cout << "\n\n intrinsic:-\n" << intrinsic;
    cout << "\n\n distCoeffs:-\n" << distCoeffs;
    copy(rvecs.begin(), rvecs.end(), ostream_iterator<Mat>(cout, "\n\n Rotation vector:-\n "));
    copy(tvecs.begin(), tvecs.end(), ostream_iterator<Mat>(cout, "\n\n Translational vector:-\n"));
    printf("\n\nDone Calibration.........!\n\n");
    
    Mat imageUndistorted;

    cap >> image;
    Mat Rot, Tran;
    int cubed = 0;
    while(1)
    {
        undistort(image, imageUndistorted, intrinsic, distCoeffs);
        vector<Point2f> corners;
        vector< Point2f > imagePoints3;
        cvtColor ( image,grayImage, COLOR_BGR2GRAY );
        bool patternfound = findChessboardCorners(grayImage, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        if(patternfound)
        {
            solvePnP(objects, corners, intrinsic, distCoeffs, Rot, Tran);
            drawAxes(imageUndistorted,intrinsic, distCoeffs, Rot, Tran,3*square_size);
        }
        
        if(clock() - prevTimeStamp > 500*1e-3*CLOCKS_PER_SEC) {
            cubed++;
            imwrite("./"+to_string(numOfBoards)+"-2/image_cubed"+to_string(cubed)+".jpg", imageUndistorted);
            prevTimeStamp = clock();
        }
        
        imshow("Undistorted Image", imageUndistorted);
        cap >> image;
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    
    return 0;
}
