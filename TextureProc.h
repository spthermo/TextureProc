#pragma once

#define EXPORT __declspec(dllexport)
#define APICALL __cdecl

#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <vector>
#include <cmath>
#include "Helper.h"

typedef unsigned char byte;

extern "C"
{
	//Receives a byte array with BGR(A) image data (1920x1080) and returns a byte array with the post-processed 
	//image ands a float array with the translation, rotation, and principal point parameters.
	EXPORT bool principal_point_localization(const byte* imgData, int width, int height, byte* imgResData, float* res, int size, bool alpha);

	//Receives a byte array with BGR(A) image data (1920x1080) and a byte array with depth image data. Returns two byte arrays with the generated hmd on color amd depth data.
	EXPORT bool artificial_hmd_placement(int device_id, const byte* imgColorData, const byte* imgDepthData, byte* imgDepthResData, byte* imgBGRResData, float* params, bool alpha);

	//Displays the post-processed BGR(A) image (mostly used for debugging)
	EXPORT void showImg(cv::Mat img);

	//Saves the post-processed BGR(A) image (mostly used for debugging)
	EXPORT void saveImg(char *name, cv::Mat img);

	//Colorization of the depth image (mostly used for debugging)
	EXPORT void colorizeDepth(cv::Mat imgd);
	
	//Rotates 3D points
	EXPORT void rotate(double pitch, double roll, double yaw, std::vector<cv::Point3d> &points);
}
