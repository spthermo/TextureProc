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

typedef unsigned char byte;

extern "C"
{
	//Receives a byte array with BGRA image data (1920x1080) and returns a byte array with the post-processed 
	//BGRA image (HMD noise rectangle), as well as a float array with the translation, rotation, and bbox parameters.
	EXPORT bool Bbox_BGRA(const byte* imgData, int width, int height, byte* imgResData, float* res, int size);

	//Receives a byte array with BGR image data (1920x1080) and returns a byte array with the post-processed 
	//BGR image (HMD noise rectangle), as well as a float array with the translation, rotation, and bbox parameters.
	EXPORT bool Bbox_BGR(const byte* imgData, int width, int height, byte* imgResData, float* res, int size);

	//Receives a byte array with BGRA image data (1920x1080) and returns a byte array with the hmd-mapped Depth data
	EXPORT bool BGRA2depth(const byte* imgColorData, const byte* imgDepthData, byte* imgResData, float* params);

	//Receives a byte array with BGR image data (1920x1080) and returns a byte array with the hmd-mapped Depth data
	EXPORT bool BGR2depth(const byte* imgColorData, const byte* imgDepthData, byte* imgResData, float* params);
	
	//Displays the post-processed BGR/BGRA image (mostly used for debugging)
	EXPORT void ShowImg(cv::Mat img);

	//Saves the post-processed BGR/BGRA image (mostly used for debugging)
	EXPORT void SaveImg(char *name, cv::Mat img);

	//Colorization of the depth image (mostly used for debugging)
	EXPORT void ColorizeDepth(cv::Mat imgd);

	//Computes the intersection between two diagonals of the HMD and returns the intersection coordinates as a cv::Point
	EXPORT cv::Point ComputePP(cv::Point** front, cv::Point** back);
	
	//Rotates 3D points
	EXPORT void Rotate(double pitch, double roll, double yaw, std::vector<cv::Point3d> &points);
}