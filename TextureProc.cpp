#include "TextureProc.h"

//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

static std::unique_ptr<std::vector<double>> face_params(new std::vector<double>());

//Displays the post-processed BGR(A) image (mostly used for debugging)
EXPORT void showImg(cv::Mat img)
{
	cv::imshow("Image", img);
	unsigned char key = cv::waitKey(0);
	cv::waitKey(0);
}

//Saves the post-processed BGR(A) image (mostly used for debugging)
EXPORT void saveImg(char *name, cv::Mat img)
{
	cv::imwrite(name, img);
}

//Colorization of the depth image (mostly used for debugging)
EXPORT void colorizeDepth(cv::Mat imgd)
{
	double min, max;
	cv::Mat adjMap;
	cv::Mat colorMap; // = cv::Mat::zeros(424, 512, CV_8UC3);
	cv::minMaxIdx(imgd, &min, &max);
	cv::convertScaleAbs(imgd, adjMap);
	imgd.convertTo(adjMap, CV_8UC1, 0.15);
	cv::applyColorMap(adjMap, colorMap, cv::COLORMAP_JET);
	//cv::imwrite("final_depth.png", colorMap);
	//showImg(colorMap);
}

//Receives a byte array with BGR(A) image data (1920x1080) and returns a byte array with the post-processed 
//image ands a float array with the translation, rotation, and principal point parameters.
EXPORT bool principal_point_localization(const byte* input, int width, int height, byte* imgResData, float* res, int size, bool alpha)
{
	bool success = false;
	std::ofstream log("mylog.txt");
	
	//Byte array to cv Mat
	auto bytesize = (alpha) ? 4 * width * height : 3 * width * height;

	std::vector<byte> input_image(input, input + bytesize);
	cv::Mat img = (alpha) ? cv::Mat(height, width, CV_8UC4) : cv::Mat(height, width, CV_8UC3);	
	std::memcpy(img.data, input_image.data(), input_image.size());
	
	std::vector<float> ptrR, ptrT;
	std::vector<float> bbox, noise, pp;
	std::vector<dlib::rectangle> faces;

	//Load face detection and pose estimation models (dlib).
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;

	try
	{
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
	}
	catch (std::exception e)
	{
		log << e.what() << std::endl;
	}

	//Fill in cam intrinsics and distortion coefficients
	cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
	cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

	//Fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
	std::vector<cv::Point3d> object_pts;
	object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
	object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
	object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
	object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
	object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
	object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
	object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
	object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
	object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
	object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
	object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
	object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
	object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
	object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

																		 //2D ref points(image coordinates), referenced from detected facial feature
	std::vector<cv::Point2d> image_pts;

	//Result
	cv::Mat rotation_vec;                           //3 x 1
	cv::Mat rotation_mat;                           //3 x 3 R
	cv::Mat translation_vec;                        //3 x 1 T
	cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
	cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

	//Reproject 3D points world coordinate axis to verify result pose
	std::vector<cv::Point3d> reprojectsrc;

	reprojectsrc.push_back(cv::Point3d(0.5, 7.0, 6.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 7.0, 5.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.0, 5.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.0, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.0, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.0, 5.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.0, 5.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.0, 6.0));

	//Reprojected 2D points
	std::vector<cv::Point2d> reprojectdst;
	reprojectdst.resize(8);

	//Temp buf for decomposeProjectionMatrix()
	cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
	cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
	cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

	//Text on screen
	std::ostringstream outtext;
	cv::Rect rect_old(100, 100, 150, 150);

	std::vector<cv::Rect> init_faces;
	cv::Mat temp, temp_gray, croppedImg;
	if (alpha)
	{
		cv::cvtColor(img, temp, CV_BGRA2BGR);
	}
	else
	{
		temp = img;
	}

	try
	{
		face_cascade.load(face_cascade_name);
		cv::cvtColor(img, temp_gray, CV_BGRA2GRAY);
		cv::equalizeHist(temp_gray, temp_gray);

		//Face detection
		face_cascade.detectMultiScale(temp_gray, init_faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
		if (init_faces.size() < 1)
		{
			log << "No face detected!!" << std::endl;
			return success;
		}
		else if (init_faces.size() == 1) 
		{
			if (init_faces[0].width > 50 && init_faces[0].width < 250)
			{
				cv::Point center(init_faces[0].x + init_faces[0].width*0.5, init_faces[0].y + init_faces[0].height*0.5);
				cv::Rect rect(init_faces[0].x - 100, init_faces[0].y - 100, init_faces[0].width + 200, init_faces[0].height + 200);
				if (rect.x >= 0 && rect.y >= 0 && rect.width >= 0 && rect.height >= 0 && rect.width + rect.x < img.cols && rect.height + rect.y < img.rows)
				{
					croppedImg = temp(rect);
					rect_old = rect;
				}
				else
				{
					log << "ROI should have non-negative values!!" << std::endl;
					return success;
				}
			}
			else 
			{
				log << "Face size not compatible!!" << std::endl;
				return success;
			}

			dlib::cv_image<dlib::bgr_pixel> cimg(croppedImg);

			//Detect faces
			faces = detector(cimg);
			if (faces.size() != 1)
			{
				log << "DLIB detector could not fit on detected face!!" << std::endl;
				return success;
			}

			//Track features
			dlib::full_object_detection shape = predictor(cimg, faces[0]);

			//Fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
			image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
			image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
			image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
			image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
			image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
			image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
			image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
			image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
			image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
			image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
			image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
			image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
			image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
			image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

																					  //Calculate pose
			cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
			std::cout << "translation_vec = " << std::endl << " " << translation_vec << std::endl << std::endl;
			std::cout << "rotation_vec = " << std::endl << " " << rotation_vec << std::endl << std::endl;

			//Reproject
			cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);
			int npt[] = { 4 };

			//Create the HMD rectangle on the original image
			cv::Point hmd_frontal[4];
			cv::Point hmd_back[4];
			cv::Point hmd_left[4];
			cv::Point hmd_right[4];
			cv::Point hmd_top[4];
			cv::Point hmd_bottom[4];

			//Frontal side
			hmd_frontal[0].x = rect_old.tl().x + reprojectdst[0].x;
			hmd_frontal[0].y = rect_old.tl().y + reprojectdst[0].y;

			hmd_frontal[1].x = rect_old.br().x - (rect_old.width - reprojectdst[4].x);
			hmd_frontal[1].y = rect_old.tl().y + reprojectdst[4].y;

			hmd_frontal[2].x = rect_old.br().x - (rect_old.width - reprojectdst[7].x);
			hmd_frontal[2].y = rect_old.br().y - (rect_old.width - reprojectdst[7].y);

			hmd_frontal[3].x = rect_old.tl().x + reprojectdst[3].x;
			hmd_frontal[3].y = rect_old.br().y - (rect_old.width - reprojectdst[3].y);

			//Back side
			hmd_back[0].x = rect_old.tl().x + reprojectdst[1].x;
			hmd_back[0].y = rect_old.tl().y + reprojectdst[1].y;

			hmd_back[1].x = rect_old.br().x - (rect_old.width - reprojectdst[5].x);
			hmd_back[1].y = rect_old.tl().y + reprojectdst[5].y;

			hmd_back[2].x = rect_old.br().x - (rect_old.width - reprojectdst[6].x);
			hmd_back[2].y = rect_old.br().y - (rect_old.width - reprojectdst[6].y);

			hmd_back[3].x = rect_old.tl().x + reprojectdst[2].x;
			hmd_back[3].y = rect_old.br().y - (rect_old.width - reprojectdst[2].y);

			//Left side
			hmd_left[0].x = rect_old.tl().x + reprojectdst[1].x;
			hmd_left[0].y = rect_old.tl().y + reprojectdst[1].y;

			hmd_left[1].x = rect_old.br().x - (rect_old.width - reprojectdst[0].x);
			hmd_left[1].y = rect_old.tl().y + reprojectdst[0].y;

			hmd_left[2].x = rect_old.br().x - (rect_old.width - reprojectdst[3].x);
			hmd_left[2].y = rect_old.br().y - (rect_old.width - reprojectdst[3].y);

			hmd_left[3].x = rect_old.tl().x + reprojectdst[2].x;
			hmd_left[3].y = rect_old.br().y - (rect_old.width - reprojectdst[2].y);

			//Right side
			hmd_right[0].x = rect_old.tl().x + reprojectdst[5].x;
			hmd_right[0].y = rect_old.tl().y + reprojectdst[5].y;

			hmd_right[1].x = rect_old.br().x - (rect_old.width - reprojectdst[4].x);
			hmd_right[1].y = rect_old.tl().y + reprojectdst[4].y;

			hmd_right[2].x = rect_old.br().x - (rect_old.width - reprojectdst[7].x);
			hmd_right[2].y = rect_old.br().y - (rect_old.width - reprojectdst[7].y);

			hmd_right[3].x = rect_old.tl().x + reprojectdst[6].x;
			hmd_right[3].y = rect_old.br().y - (rect_old.width - reprojectdst[6].y);

			//Top side
			hmd_top[0].x = rect_old.tl().x + reprojectdst[3].x;
			hmd_top[0].y = rect_old.tl().y + reprojectdst[3].y;

			hmd_top[1].x = rect_old.br().x - (rect_old.width - reprojectdst[7].x);
			hmd_top[1].y = rect_old.tl().y + reprojectdst[7].y;

			hmd_top[2].x = rect_old.br().x - (rect_old.width - reprojectdst[6].x);
			hmd_top[2].y = rect_old.br().y - (rect_old.width - reprojectdst[6].y);

			hmd_top[3].x = rect_old.tl().x + reprojectdst[2].x;
			hmd_top[3].y = rect_old.br().y - (rect_old.width - reprojectdst[2].y);

			//Bottom side
			hmd_bottom[0].x = rect_old.tl().x + reprojectdst[0].x;
			hmd_bottom[0].y = rect_old.tl().y + reprojectdst[0].y;

			hmd_bottom[1].x = rect_old.br().x - (rect_old.width - reprojectdst[4].x);
			hmd_bottom[1].y = rect_old.tl().y + reprojectdst[4].y;

			hmd_bottom[2].x = rect_old.br().x - (rect_old.width - reprojectdst[5].x);
			hmd_bottom[2].y = rect_old.br().y - (rect_old.width - reprojectdst[5].y);

			hmd_bottom[3].x = rect_old.tl().x + reprojectdst[1].x;
			hmd_bottom[3].y = rect_old.br().y - (rect_old.width - reprojectdst[1].y);

			//Drawing
			cv::Scalar scalar = (alpha) ? cv::Scalar(255, 0 ,255, 255) : cv::Scalar(255, 0, 255);

			const cv::Point* ppt[1] = { hmd_frontal };
			fillPoly(img,
				ppt,
				npt,
				1,
				scalar,
				8);

			const cv::Point* ppt2[1] = { hmd_back };
			fillPoly(img,
				ppt2,
				npt,
				1,
				scalar,
				8);

			const cv::Point* ppt3[1] = { hmd_left };
			fillPoly(img,
				ppt3,
				npt,
				1,
				scalar,
				8);

			const cv::Point* ppt4[1] = { hmd_right };
			fillPoly(img,
				ppt4,
				npt,
				1,
				scalar,
				8);

			const cv::Point* ppt5[1] = { hmd_top };
			fillPoly(img,
				ppt5,
				npt,
				1,
				scalar,
				8);

			const cv::Point* ppt6[1] = { hmd_bottom };
			fillPoly(img,
				ppt6,
				npt,
				1,
				scalar,
				8);
			
			//Calculate euler angle
			cv::Rodrigues(rotation_vec, rotation_mat);
			cv::hconcat(rotation_mat, translation_vec, pose_mat);
			cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

			//Assign the rotation values
			face_params->assign(rotation_vec.begin<double>(), rotation_vec.end<double>());

			//Assign the translation values
			ptrT.assign(translation_vec.begin<double>(), translation_vec.end<double>());
			face_params->insert(face_params->end(), ptrT.begin(), ptrT.end());

			//Assign the bbox coords
			bbox.push_back((float)rect_old.tl().x);
			bbox.push_back((float)rect_old.tl().y);
			bbox.push_back((float)rect_old.br().x);
			bbox.push_back((float)rect_old.tl().y);
			bbox.push_back((float)rect_old.br().x);
			bbox.push_back((float)rect_old.br().y);
			bbox.push_back((float)rect_old.tl().x);
			bbox.push_back((float)rect_old.br().y);
			bbox.push_back((float)rect_old.width);
			bbox.push_back((float)rect_old.height);
			face_params->insert(face_params->end(), bbox.begin(), bbox.end());

			//Assign HMD rectangle coords on the original image
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_frontal[i].x);
				noise.push_back((float)hmd_frontal[i].y);
			}
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_back[i].x);
				noise.push_back((float)hmd_back[i].y);
			}
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_left[i].x);
				noise.push_back((float)hmd_left[i].y);
			}
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_right[i].x);
				noise.push_back((float)hmd_right[i].y);
			}
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_top[i].x);
				noise.push_back((float)hmd_top[i].y);
			}
			for (int i = 0; i < 4; i++)
			{
				noise.push_back((float)hmd_bottom[i].x);
				noise.push_back((float)hmd_bottom[i].y);
			}
			face_params->insert(face_params->end(), noise.begin(), noise.end());

			//The serialized vector contains (62 elements)
			//[0-2] rotation 
			//[3-5] translation
			//[6-15] face bbox 
			//[16-61] noise (hmd) box on the original image [front]->[back]->[left]->[right]->[top]->[bottom]
			for (int i = 0; i < size; i++)
			{
				res[i] = (float)face_params->at(i);
				//std::cout << res[i] << std::endl;
			}

			//showImg(img);
			//Post-processing cv Mat to byte array
			std::memcpy(imgResData, img.data, img.total() * img.elemSize() * sizeof(byte));

			//showImg(img);
			success = true;
			log.close();

			return success;
		}
		else 
		{
			log << "Multiple faces detected!!" << std::endl;
			return success;
		}
	}
	catch (std::exception e)
	{
		log << e.what();
		log.close();

		return success;
	}
}

//Receives a byte array with BGR(A) image data (1920x1080) and a byte array with depth image data. Returns two byte arrays with the generated hmd on color amd depth data.
EXPORT bool artificial_hmd_placement(int device_id, const byte* imgColorData, int colorWidth, int colorHeight, const byte* imgDepthData, int depthWidth, int depthHeight, byte* imgDepthResData, byte* imgBGRResData, float* params, bool alpha)
{
	bool success = false;
	uchar channel_b, channel_g, channel_r, channel_a;
	float depthValue;

	//D10 sensor parameters
	std::vector< std::vector<float>> R(3, std::vector<float>(3, 0));
	std::vector<float> T(3, 0);
	std::vector< std::vector<float>> cam_params(2, std::vector<float>(4, 0));

	std::vector< std::vector<float>> P3D(3, std::vector<float>(1, 0));
	std::vector< std::vector<float>> P3D_new(3, std::vector<float>(1, 0));
	std::vector< std::vector<int>> P2D(2, std::vector<int>(1, 0));

	std::ostringstream oss;
	oss << "KRT" << device_id << ".txt";
	// const char* krt_file = oss.str().c_str();
	read_ext_file(oss.str().c_str(), R, T, cam_params);

	//Focal and distortion params
	float fx_d = cam_params[0][0];
	float fy_d = -cam_params[0][1];
	float cx_d = cam_params[0][2];
	float cy_d = cam_params[0][3];
	float fx_rgb = cam_params[1][0];
	float fy_rgb = cam_params[1][1];
	float cx_rgb = cam_params[1][2];
	float cy_rgb = cam_params[1][3];

	//Byte array to cv Mat - BGR
	auto bytesize = (alpha) ? 4 * colorHeight * colorWidth : 3 * colorHeight * colorWidth;

	std::vector<byte> input_img(imgColorData, imgColorData + bytesize);
	cv::Mat img = (alpha) ? cv::Mat::zeros(colorHeight, colorWidth, CV_8UC4) : cv::Mat::zeros(colorHeight, colorWidth, CV_8UC3);
	std::memcpy(img.data, input_img.data(), input_img.size());

	//Byte array to cv Mat - Depthmap
	auto d_bytesize = 2 * 424 * depthWidth;
	std::vector<byte> input_imgd(imgDepthData, imgDepthData + d_bytesize);
	cv::Mat imgd = cv::Mat::zeros(depthHeight, depthWidth, CV_16U);
	cv::Mat imgd_rgb = cv::Mat::zeros(depthHeight, depthWidth, CV_8UC3);
	std::memcpy(imgd.data, input_imgd.data(), input_imgd.size());

	//For visualization purposes
	cv::Mat colorized_imgd = cv::Mat::zeros(depthHeight, depthWidth, CV_8UC3);

	cv::Vec4b ColorValue_bgra;
	cv::Vec3b ColorValue_bgr;

	cv::imwrite("test_color.png", img);
	cv::imwrite("depth_original.png", imgd);

	std::vector<ushort> box_depthvalues;
	std::vector<int> box_x_values;
	std::vector<int> box_y_values;

	//Depth to color mapping
	for (int i = 0; i < imgd.cols; i++)
	{
		for (int j = 0; j < imgd.rows; j++)
		{
			depthValue = imgd.at<ushort>(j, i);

			if (depthValue > 0)
			{
				P3D[0][0] = (i - cx_d) * depthValue / fx_d;
				P3D[1][0] = (j - cy_d) * depthValue / fy_d;

				P3D[2][0] = depthValue;

				for (int k = 0; k < 3; k++)
				{
					P3D_new[k][0] = (R[k][0] * P3D[0][0]) + (R[k][1] * P3D[1][0]) + (R[k][2] * P3D[2][0]) + T[k];
				}

				P2D[0][0] = (int)(P3D_new[0][0] * fx_rgb) / P3D_new[2][0] + cx_rgb;
				P2D[1][0] = (int)(P3D_new[1][0] * fy_rgb) / P3D_new[2][0] + cy_rgb;

				if (P2D[0][0] > 0 && P2D[0][0] < colorWidth && P2D[1][0] > 0 && P2D[1][0] < colorHeight)
				{
					if (alpha)
					{
						ColorValue_bgra = img.at<cv::Vec4b>(P2D[1][0], P2D[0][0]);
						channel_b = ColorValue_bgra.val[0];
						channel_g = ColorValue_bgra.val[1];
						channel_r = ColorValue_bgra.val[2];
						channel_a = ColorValue_bgra.val[3];
					}
					else
					{
						ColorValue_bgr = img.at<cv::Vec3b>(P2D[1][0], P2D[0][0]);
						channel_b = ColorValue_bgr.val[0];
						channel_g = ColorValue_bgr.val[1];
						channel_r = ColorValue_bgr.val[2];
					}

					//Map principal point from BGR(A) to depth
					if (channel_b == 255 && channel_g == 0 && channel_r == 255)
					{
						imgd.at<ushort>(j, i) = 65000;

						if (depthValue > 50 && depthValue < 3500)
						{
							box_x_values.push_back(i);
							box_y_values.push_back(j);
							box_depthvalues.push_back(depthValue);
						}
					}
				}
			}
		}
	}

	// showImg(colorized_imgd);

	auto med_depth_value = (ushort)CalcMedian<ushort>(box_depthvalues);
	auto med_y_value = (int)CalcMedian<int>(box_y_values);
	auto med_x_value = (int)CalcMedian<int>(box_x_values);

	cv::Point3d central_3d_point = cv::Point3d(
		(med_x_value - cx_d) * med_depth_value / fx_d,
		(med_y_value - cy_d) * med_depth_value / fy_d,
		med_depth_value
	);

	// 3D Construction appropriate for rendering
	std::vector<cv::Point3d> Construction3D;

	Construction3D.clear();

	float width = 65, height = 45, depth = 45;

	for (int x = -width; x < width; ++x)
		for (int y = -height; y < height; ++y)
			for (int z = -depth; z < depth; ++z)
			{
				Construction3D.push_back(cv::Point3d(x, y, z));
			}

	rotate(params[1], params[2], params[0], Construction3D);

	auto cp_dp_x = (int)(central_3d_point.x * fx_d) / central_3d_point.z + cx_d;
	auto cp_dp_y = (int)(central_3d_point.y * fy_d) / central_3d_point.z + cy_d;

	// Masked image and 2.5D data
	cv::Mat imgd_mask = cv::Mat::zeros(depthHeight, depthWidth, CV_8U);
	std::vector<cv::Point3d> masked_points;

	for (auto i = 0; i < Construction3D.size(); ++i) {
		Construction3D[i] += central_3d_point;

		auto dp_x = (int)(Construction3D[i].x * fx_d) / Construction3D[i].z + cx_d;
		auto dp_y = (int)(Construction3D[i].y * fy_d) / Construction3D[i].z + cy_d;

		if (dp_x > 0 && dp_x < depthWidth && dp_y > 0 && dp_y < depthHeight) {
			if (imgd.at<ushort>(dp_y, dp_x) > Construction3D[i].z || imgd.at<ushort>(dp_y, dp_x) < 50)
			{
				masked_points.push_back(cv::Point3d(dp_x, dp_y, Construction3D[i].z));
				imgd.at<ushort>(dp_y, dp_x) = Construction3D[i].z;
				imgd_mask.at<byte>(dp_y, dp_x) = 255;
			}
		}
	}

	//showImg(imgd_mask);
	std::vector<cv::Point> mask_points, conv_hull;
	cv::Mat img_rgb_mask = cv::Mat::zeros(colorHeight, colorWidth, CV_8U);

#pragma region Custom Concave Hull
	for (int j = 0; j < imgd_mask.rows; j++)
	{
		bool mask_point_found_i = false;
		int first_mask_point_found_i = 0;

		for (int i = 0; i < imgd_mask.cols; i++)
		{
			if (imgd_mask.at<byte>(j, i) == 255)
			{
				P3D[0][0] = (i - cx_d) * imgd.at<ushort>(j, i) / fx_d;
				P3D[1][0] = (j - cy_d) * imgd.at<ushort>(j, i) / fy_d;

				P3D[2][0] = imgd.at<ushort>(j, i);

				for (int k = 0; k < 3; k++)
				{
					P3D_new[k][0] = (R[k][0] * P3D[0][0]) + (R[k][1] * P3D[1][0]) + (R[k][2] * P3D[2][0]) + T[k];
				}

				P2D[0][0] = (int)(P3D_new[0][0] * fx_rgb) / P3D_new[2][0] + cx_rgb;
				P2D[1][0] = (int)(P3D_new[1][0] * fy_rgb) / P3D_new[2][0] + cy_rgb;

				if (P2D[0][0] > 0 && P2D[0][0] < colorWidth && P2D[1][0] > 0 && P2D[1][0] < colorHeight)
				{
					auto dist_i = P2D[0][0] - first_mask_point_found_i;

					if (mask_point_found_i && dist_i < 8 && dist_i > 0)
					{
						for (int k = first_mask_point_found_i; k < P2D[0][0] + 1; ++k)
						{
							img_rgb_mask.at<byte>(P2D[1][0], k) = 255;
						}
						first_mask_point_found_i = P2D[0][0];
					}
					else
					{
						img_rgb_mask.at<byte>(P2D[1][0], P2D[0][0]) = 255;
						first_mask_point_found_i = P2D[0][0];
						mask_point_found_i = true;
					}		
				}
			}
		}
	}

	//Depth to color mapping
	for (int i = 0; i < img_rgb_mask.cols; i++)
	{
		bool mask_point_found_j = false;
		int first_mask_point_found_j = 0;

		for (int j = 0; j < img_rgb_mask.rows; j++)
		{
			if (img_rgb_mask.at<byte>(j, i) == 255)
			{				
				auto dist_j = j - first_mask_point_found_j;

				if (mask_point_found_j && dist_j < 8 && dist_j > 0)
				{
					for (int k = first_mask_point_found_j; k < j + 1; ++k)
					{
						img_rgb_mask.at<byte>(k, i) = 255;
					}
					first_mask_point_found_j = j;
				}
				else
				{
					img_rgb_mask.at<byte>(j, i) = 255;
					first_mask_point_found_j = j;
					mask_point_found_j = true;
				}				
			}
		}
	}

#pragma endregion Custom Concave Hull Concave Hull // this should be replace by a more sophisticated CCH alforithm (e.g. https://www.codeproject.com/Articles/1201438/The-Concave-Hull-of-a-Set-of-Points)

	std::vector< std::vector< cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(img_rgb_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	
	//Drawing
	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		cv::drawContours(img, contours, i, cv::Scalar(25, 35, 25), -1, 8, hierarchy);
	}

	//saveImg("final_texture.jpg", img);
	//saveImg("final_mask_d.jpg", imgd_mask);
	//showImg(colorized_imgd);
	//showImg(img);
	colorizeDepth(imgd);

	//Post-processing cv Mat to byte array
	std::memcpy(imgDepthResData, imgd.data, imgd.total() * imgd.elemSize() * sizeof(byte));
	std::memcpy(imgBGRResData, img.data, img.total() * img.elemSize() * sizeof(byte));

	success = true;

	return success;
}

//Rotates 3D points
EXPORT void rotate(double pitch, double roll, double yaw, std::vector<cv::Point3d> &points) {
	auto cosa = std::cos(yaw);
	auto sina = std::sin(yaw);

	auto cosb = std::cos(pitch);
	auto sinb = std::sin(pitch);

	auto cosc = std::cos(roll);
	auto sinc = std::sin(roll);

	auto Axx = cosa*cosb;
	auto Axy = cosa*sinb*sinc - sina*cosc;
	auto Axz = cosa*sinb*cosc + sina*sinc;

	auto Ayx = sina*cosb;
	auto Ayy = sina*sinb*sinc + cosa*cosc;
	auto Ayz = sina*sinb*cosc - cosa*sinc;

	auto Azx = -sinb;
	auto Azy = cosb*sinc;
	auto Azz = cosb*cosc;

	for (auto i = 0; i < points.size(); i++) {
		auto px = points[i].x;
		auto py = points[i].y;
		auto pz = points[i].z;

		points[i].x = Axx*px + Axy*py + Axz*pz;
		points[i].y = Ayx*px + Ayy*py + Ayz*pz;
		points[i].z = Azx*px + Azy*py + Azz*pz;
	}
}

