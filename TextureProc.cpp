#include "TextureProc.h"

//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

static std::unique_ptr<std::vector<double>> face_params(new std::vector<double>());



//Displays the post-processed BGR/BGRA image (mostly used for debugging)
EXPORT void ShowImg(cv::Mat img)
{
	cv::imshow("Image", img);
	unsigned char key = cv::waitKey(0);
	cv::waitKey(0);
}

//Saves the post-processed BGR/BGRA image (mostly used for debugging)
EXPORT void SaveImg(char *name, cv::Mat img)
{
	cv::imwrite(name, img);
}

//Colorization of the depth image (mostly used for debugging)
EXPORT void ColorizeDepth(cv::Mat imgd)
{
	double min, max;
	cv::Mat adjMap;
	cv::Mat colorMap; // = cv::Mat::zeros(424, 512, CV_8UC3);
	cv::minMaxIdx(imgd, &min, &max);
	cv::convertScaleAbs(imgd, adjMap);
	imgd.convertTo(adjMap, CV_8UC1, 0.15);
	cv::applyColorMap(adjMap, colorMap, cv::COLORMAP_JET);
	cv::imwrite("depth.png", colorMap);
	ShowImg(colorMap);
}

//Computes the intersection between two diagonals of the HMD and returns the intersection coordinates as a cv::Point
EXPORT cv::Point ComputePP(cv::Point** front, cv::Point** back)
{
	cv::Point res[1];
	float a1, b1, c1, a2, b2, c2, det;

	a1 = back[0][2].y - front[0][0].y;
	b1 = front[0][0].x - back[0][2].x;
	c1 = (front[0][0].x * a1) + (front[0][0].y * b1);

	a2 = back[0][3].y - front[0][1].y;
	b2 = front[0][1].x - back[0][3].x;
	c2 = (front[0][1].x * a2) + (front[0][1].y * b2);

	det = (a1 * b2) - (a2 * b1);

	res[0].x = static_cast<float>((c1 * b2) - (c2 * b1)) / det;
	res[0].y = static_cast<float>((c2 * a1) - (c1 * a2)) / det;

	return res[0];
}

//Receives a byte array with BGRA image data (1920x1080) and returns a byte array with the post-processed 
//BGRA image (HMD noise rectangle), as well as a float array with the translation, rotation, and bbox parameters.
EXPORT bool Bbox_BGRA(const byte* input, int width, int height, byte* imgResData, float* res, int size)
{
	bool success = false;
	std::ofstream log("mylog.txt");

	//Byte array to cv Mat
	auto bytesize = 4 * width * height;
	std::vector<byte> input_image(input, input + bytesize);
	cv::Mat img = cv::Mat(height, width, CV_8UC4);
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

	reprojectsrc.push_back(cv::Point3d(0.5, 7.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 7.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.5, 7.0));

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
	cv::cvtColor(img, temp, CV_BGRA2BGR);
	try
	{
		face_cascade.load(face_cascade_name);
		cv::cvtColor(temp, temp_gray, CV_BGRA2GRAY);
		cv::equalizeHist(temp_gray, temp_gray);
		//SaveImg("temp_gray.png", temp_gray);

		//Face detection
		face_cascade.detectMultiScale(temp_gray, init_faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
		if (init_faces.size() < 1)
		{
			log << "No face detected!!" << std::endl;
		}
		else {
			for (size_t i = 0; i < init_faces.size(); i++)
			{
				if (init_faces[i].width > 50 && init_faces[i].width < 250)
				{
					cv::Point center(init_faces[i].x + init_faces[i].width*0.5, init_faces[i].y + init_faces[i].height*0.5);
					cv::Rect rect(init_faces[i].x - 100, init_faces[i].y - 100, init_faces[i].width + 200, init_faces[i].height + 200);
					//cv::rectangle(temp, rect, cv::Scalar(255, 0, 0));
					croppedImg = temp(rect);
					rect_old = rect;
				}
				else
				{
					croppedImg = temp(rect_old);
				}
			}

			//ShowImg(croppedImg);
			dlib::cv_image<dlib::bgr_pixel> cimg(croppedImg);

			//Detect faces
			faces = detector(cimg);
			if (faces.size() == 0)
			{
				log << "dlib error: no face detected!!";
				log.close();
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
			const cv::Point* ppt[1] = { hmd_frontal };
			fillPoly(img,
				ppt,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt2[1] = { hmd_back };
			fillPoly(img,
				ppt2,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt3[1] = { hmd_left };
			fillPoly(img,
				ppt3,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt4[1] = { hmd_right };
			fillPoly(img,
				ppt4,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt5[1] = { hmd_top };
			fillPoly(img,
				ppt5,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt6[1] = { hmd_bottom };
			fillPoly(img,
				ppt6,
				npt,
				1,
				cv::Scalar(255, 0, 255),
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

			//Post-processing cv Mat to byte array
			std::memcpy(imgResData, img.data, img.total() * img.elemSize() * sizeof(byte));
			//ShowImg(img);
			success = true;
			log.close();
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

//Receives a byte array with BGR image data (1920x1080) and returns a byte array with the post-processed 
//BGR image (HMD noise rectangle), as well as a float array with the translation, rotation, and bbox parameters.
EXPORT bool Bbox_BGR(const byte* input, int width, int height, byte* imgResData, float* res, int size)
{
	bool success = false;
	std::ofstream log("mylog2.txt");

	//Byte array to cv Mat
	auto bytesize = 3 * width * height;
	std::vector<byte> input_image(input, input + bytesize);
	cv::Mat img = cv::Mat(height, width, CV_8UC3);
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

	reprojectsrc.push_back(cv::Point3d(0.5, 7.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 7.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(0.5, 6.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.5, 7.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 7.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.5, 6.0));
	reprojectsrc.push_back(cv::Point3d(-0.5, 6.5, 7.0));

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
	temp = img;

	try
	{
		face_cascade.load(face_cascade_name);
		cv::cvtColor(temp, temp_gray, CV_BGR2GRAY);
		cv::equalizeHist(temp_gray, temp_gray);
		//SaveImg("temp_gray.png", temp_gray);

		//Face detection
		face_cascade.detectMultiScale(temp_gray, init_faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
		if (init_faces.size() < 1)
		{
			log << "No face detected!!" << std::endl;
		}
		else {
			for (size_t i = 0; i < init_faces.size(); i++)
			{
				if (init_faces[i].width > 50 && init_faces[i].width < 250)
				{
					cv::Point center(init_faces[i].x + init_faces[i].width*0.5, init_faces[i].y + init_faces[i].height*0.5);
					cv::Rect rect(init_faces[i].x - 100, init_faces[i].y - 100, init_faces[i].width + 200, init_faces[i].height + 200);
					//cv::rectangle(temp, rect, cv::Scalar(255, 0, 0));
					croppedImg = temp(rect);
					rect_old = rect;
				}
				else
				{
					croppedImg = temp(rect_old);
				}
			}
			// ShowImg(croppedImg);
			dlib::cv_image<dlib::bgr_pixel> cimg(croppedImg);

			//Detect faces
			faces = detector(cimg);

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
			const cv::Point* ppt[1] = { hmd_frontal };
			fillPoly(img,
				ppt,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt2[1] = { hmd_back };
			fillPoly(img,
				ppt2,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt3[1] = { hmd_left };
			fillPoly(img,
				ppt3,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt4[1] = { hmd_right };
			fillPoly(img,
				ppt4,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt5[1] = { hmd_top };
			fillPoly(img,
				ppt5,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			const cv::Point* ppt6[1] = { hmd_bottom };
			fillPoly(img,
				ppt6,
				npt,
				1,
				cv::Scalar(255, 0, 255),
				8);

			//Compute principal point of the HMD "volume"
			cv::Point* p1[1] = { hmd_frontal };
			cv::Point* p2[1] = { hmd_back };
			cv::Point principal_point[1];

			principal_point[0] = ComputePP(p1, p2);

			std::cout << "Principal point coords (x,y): " << principal_point[0].x << " , " << principal_point[0].y << std::endl;
			//For visualization of the principal point uncomment the next 5 lines
			//cv::line(img, hmd_frontal[0], hmd_back[2], cv::Scalar(255, 255, 0), 2);
			//cv::line(img, hmd_frontal[1], hmd_back[3], cv::Scalar(255, 255, 0), 2);
			//cv::circle(img, principal_point[0], 5, cv::Scalar(0, 0, 255), 3);
			//ShowImg(img);

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

			//Post-processing cv Mat to byte array
			std::memcpy(imgResData, img.data, img.total() * img.elemSize() * sizeof(byte));

			ShowImg(img);
			success = true;
			log.close();

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

//Receives a byte array with BGRA image data (1920x1080) and returns a byte array with the hmd-mapped Depth data
EXPORT bool BGRA2depth(const byte* imgColorData, const byte* imgDepthData, byte* imgResData, float* params)
{
	bool success = false;
	uchar channel_b, channel_g, channel_r;
	float depthValue;

	//D10 sensor parameters
	std::vector< std::vector<float>> R(3, std::vector<float>(3, 0));
	std::vector< std::vector<float>> T(3, std::vector<float>(1, 0));
	std::vector< std::vector<float>> P3D(3, std::vector<float>(1, 0));
	std::vector< std::vector<float>> P3D_new(3, std::vector<float>(1, 0));
	std::vector< std::vector<int>> P2D(2, std::vector<int>(1, 0));

	//Rotation matrix
	R[0][0] = -1;
	R[0][1] = 0.00044;
	R[0][2] = 0;
	R[1][0] = 0.00044;
	R[1][1] = 1;
	R[1][2] = 0;
	R[2][0] = 0;
	R[2][1] = 0;
	R[2][2] = -1;

	//Translation vector
	T[0][0] = -51.57065;
	T[1][0] = 0.00001;
	T[2][0] = 0.00000;

	//Focal and distortion params
	float fy_d = 364.3239;
	float fx_d = -364.3239;
	float cy_d = 258.5376;
	float cx_d = 203.6222;
	float fx_rgb = 1090.37279;
	float fy_rgb = 1089.71454;
	float cx_rgb = 941.28091;
	float cy_rgb = 577.02934;

	//Byte array to cv Mat - BGR
	auto bytesize = 4 * 1080 * 1920;
	std::vector<byte> input_img(imgColorData, imgColorData + bytesize);
	cv::Mat img = cv::Mat::zeros(1080, 1920, CV_8UC4);
	std::memcpy(img.data, input_img.data(), input_img.size());
	cv::cvtColor(img, img, CV_BGRA2BGR);
	
	//Byte array to cv Mat - Depthmap
	auto d_bytesize = 2 * 424 * 512;
	std::vector<byte> input_imgd(imgDepthData, imgDepthData + d_bytesize);
	cv::Mat imgd = cv::Mat::zeros(424, 512, CV_16U);
	std::memcpy(imgd.data, input_imgd.data(), input_imgd.size());

	//For visualization purposes
	cv::Mat colorized_imgd = cv::Mat::zeros(424, 512, CV_8UC3);

	cv::Vec3b ColorValue;

	cv::Point3d central_3d_point;

	bool ready = false;

	cv::imwrite("depth_original.png", imgd);

	//Depth to color mapping
	for (int j = 0; j < imgd.rows; j++)
	{
		for (int i = 0; i < imgd.cols; i++)
		{
			depthValue = imgd.at<ushort>(j, i);

			if (depthValue > 0)
			{
				P3D[0][0] = (i - cy_d) * depthValue / fy_d;
				P3D[1][0] = (j - cx_d) * depthValue / fx_d;

				P3D[2][0] = depthValue;

				for (int k = 0; k < 3; k++)
				{
					P3D_new[k][0] = (R[k][0] * P3D[0][0]) + (R[k][1] * P3D[1][0]) + (R[k][2] * P3D[2][0]) + T[k][0];
				}

				P2D[0][0] = (int)(P3D_new[0][0] * fx_rgb) / P3D_new[2][0] + cx_rgb;
				P2D[1][0] = (int)(P3D_new[1][0] * fy_rgb) / P3D_new[2][0] + cy_rgb;

				if (P2D[0][0] > 0 && P2D[0][0] < 1920 && P2D[1][0] > 0 && P2D[1][0] < 1080)
				{
					ColorValue = img.at<cv::Vec3b>(P2D[1][0], P2D[0][0]);
					channel_b = ColorValue.val[0];
					channel_g = ColorValue.val[1];
					channel_r = ColorValue.val[2];

					cv::Vec3b &SetColorValue = colorized_imgd.at<cv::Vec3b>(j, i);
					SetColorValue.val[0] = channel_b;
					SetColorValue.val[1] = channel_g;
					SetColorValue.val[2] = channel_r;

					//The serialized vector contains (64 elements)
					//[0-2] rotation 
					//[3-5] translation
					//[6-15] face bbox 
					//[16-61] noise (hmd) box on the original image [front]->[back]->[left]->[right]->[top]->[bottom]
					//[62-63] principal point (x,y)

					//Set the HMD pixels to zero depth
					if (channel_b == 255 && channel_g == 0 && channel_r == 255)
					{
						imgd.at<ushort>(j, i) = 65000;

						central_3d_point = cv::Point3d(P3D[0][0], P3D[1][0], P3D[2][0]);
						ready = true;
						// central_3d_point = cv::Point3d(0, 0, P3D_new[2][0]);
						break;
					}
				}
			}
		}
		if (ready)
			break;
	}

	// 3D Construction appropriate for rendering
	std::vector<cv::Point3d> Construction3D;

	float width = 75, height = 50, depth = 45;

	for (int x = -width; x < width; ++x)
		for (int y = -height; y < height; ++y)
			for (int z = -depth; z < depth; ++z)
			{
				Construction3D.push_back(cv::Point3d(x, y, z));
			}

	Rotate(params[1], params[2], params[0], Construction3D);

	//std::cout << "p.y.r.: " << params[0] << " " << params[1] << " " << params[2] << "\n";

	//std::cout << "c.p.: " << central_3d_point.x << " " << central_3d_point.y << " " << central_3d_point.z << "\n";

	auto cp_dp_x = (int)(central_3d_point.x * fy_d) / central_3d_point.z + cy_d;
	auto cp_dp_y = (int)(central_3d_point.y * fx_d) / central_3d_point.z + cx_d;

	// imgd.at<ushort>(cp_dp_y, cp_dp_x) = 65000;

	for (auto i = 0; i < Construction3D.size(); i++) {
		Construction3D[i] += central_3d_point;

		//std::cout << Construction3D[i].x << " " << Construction3D[i].y << " " << Construction3D[i].z << "\n";

		auto dp_x = (int)(Construction3D[i].x * fy_d) / Construction3D[i].z + cy_d;
		auto dp_y = (int)(Construction3D[i].y * fx_d) / Construction3D[i].z + cx_d;

		//std::cout << dp_x << " " << dp_y << " " << Construction3D[i].z << "\n";

		//P3D[0][0] = (i - cy_d) * depthValue / fy_d;
		//P3D[1][0] = (j - cx_d) * depthValue / fx_d;
		if (imgd.at<ushort>(dp_y, dp_x) > Construction3D[i].z || imgd.at<ushort>(dp_y, dp_x) < 50)
			imgd.at<ushort>(dp_y, dp_x) = Construction3D[i].z;
	}

	// ShowImg(colorized_imgd);
	// ColorizeDepth(imgd);

	//Post-processing cv Mat to byte array
	std::memcpy(imgResData, imgd.data, imgd.total() * imgd.elemSize() * sizeof(byte));

	success = true;

	return success;
}

//Receives a byte array with BGR image data (1920x1080) and returns a byte array with the hmd-mapped Depth data
EXPORT bool BGR2depth(const byte* imgColorData, const byte* imgDepthData, byte* imgResData, float* params)
{
	bool success = false;
	uchar channel_b, channel_g, channel_r;
	float depthValue;

	//D10 sensor parameters
	std::vector< std::vector<float>> R(3, std::vector<float>(3, 0));
	std::vector< std::vector<float>> T(3, std::vector<float>(1, 0));
	std::vector< std::vector<float>> P3D(3, std::vector<float>(1, 0));
	std::vector< std::vector<float>> P3D_new(3, std::vector<float>(1, 0));
	std::vector< std::vector<int>> P2D(2, std::vector<int>(1, 0));

	//Rotation matrix // D10
	R[0][0] = -1;
	R[0][1] = 0.00044;
	R[0][2] = 0;
	R[1][0] = 0.00044;
	R[1][1] = 1;
	R[1][2] = 0;
	R[2][0] = 0;
	R[2][1] = 0;
	R[2][2] = -1;

	//Rotation matrix // D11
	R[0][0] = -0.99999;
	R[0][1] = -0.00408;
	R[0][2] = 0;
	R[1][0] = -0.00408;
	R[1][1] = 0.99999;
	R[1][2] = 0;
	R[2][0] = 0;
	R[2][1] = 0;
	R[2][2] = -1;

	//Translation vector // D10
	T[0][0] = -51.57065;
	T[1][0] = 0.00001;
	T[2][0] = 0.00000;

	//Translation vector // D11
	T[0][0] = -52.54302;
	T[1][0] = 0.00004;
	T[2][0] = 0.00000;

	//Focal and distortion params // D10
	/*float fx_d = 364.3239;
	float fy_d = -364.3239;
	float cx_d = 258.5376;
	float cy_d = 203.6222;
	float fx_rgb = 1090.37279;
	float fy_rgb = 1089.71454;
	float cx_rgb = 941.28091;
	float cy_rgb = 577.02934;*/

	//Focal and distortion params // D10
	float fx_d = 364.7277;
	float fy_d = -364.7277;
	float cx_d = 256.1403;
	float cy_d = 212.3106;
	float fx_rgb = 1070.19566;
	float fy_rgb = 1069.46443;
	float cx_rgb = 958.75354;
	float cy_rgb = 545.59161;

	//Byte array to cv Mat - BGR
	auto bytesize = 3 * 1080 * 1920;
	std::vector<byte> input_img(imgColorData, imgColorData + bytesize);
	cv::Mat img = cv::Mat::zeros(1080, 1920, CV_8UC3);
	std::memcpy(img.data, input_img.data(), input_img.size());

	//Byte array to cv Mat - Depthmap
	auto d_bytesize = 2 * 424 * 512;
	std::vector<byte> input_imgd(imgDepthData, imgDepthData + d_bytesize);
	cv::Mat imgd = cv::Mat::zeros(424, 512, CV_16U);
	cv::Mat imgd_rgb = cv::Mat::zeros(424, 512, CV_8UC3);
	std::memcpy(imgd.data, input_imgd.data(), input_imgd.size());

	//For visualization purposes
	cv::Mat colorized_imgd = cv::Mat::zeros(424, 512, CV_8UC3);

	cv::Vec3b ColorValue;


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
					P3D_new[k][0] = (R[k][0] * P3D[0][0]) + (R[k][1] * P3D[1][0]) + (R[k][2] * P3D[2][0]) + T[k][0];
				}

				P2D[0][0] = (int)(P3D_new[0][0] * fx_rgb) / P3D_new[2][0] + cx_rgb;
				P2D[1][0] = (int)(P3D_new[1][0] * fy_rgb) / P3D_new[2][0] + cy_rgb;

				//P2D[0][0] = (int)(P3D[0][0] * fx_rgb) / P3D[2][0] + cx_rgb;
				//P2D[1][0] = (int)(P3D[1][0] * fy_rgb) / P3D[2][0] + cy_rgb;

				if (P2D[0][0] > 0 && P2D[0][0] < 1920 && P2D[1][0] > 0 && P2D[1][0] < 1080)
				{
					ColorValue = img.at<cv::Vec3b>(P2D[1][0], P2D[0][0]);
					channel_b = ColorValue.val[0];
					channel_g = ColorValue.val[1];
					channel_r = ColorValue.val[2];

					cv::Vec3b &SetColorValue = colorized_imgd.at<cv::Vec3b>(j, i);
					SetColorValue.val[0] = channel_b;
					SetColorValue.val[1] = channel_g;
					SetColorValue.val[2] = channel_r;

					//The serialized vector contains (64 elements)
					//[0-2] rotation 
					//[3-5] translation
					//[6-15] face bbox 
					//[16-61] noise (hmd) box on the original image [front]->[back]->[left]->[right]->[top]->[bottom]
					//[62-63] principal point (x,y)

					//Set the HMD pixels to zero depth
					if (channel_b == 255 && channel_g == 0 && channel_r == 255)
					{
						imgd.at<ushort>(j, i) = 65000;

						if (depthValue > 50 && depthValue < 3500)
						{
							box_x_values.push_back(i);
							box_y_values.push_back(j);
							box_depthvalues.push_back(depthValue);

							// central_3d_point = cv::Point3d(P3D[0][0], P3D[1][0], 750);

							// std::cout << central_3d_point.x << " " << central_3d_point.y << " " << central_3d_point.z << std::endl;
						
						}
					}
				}
			}
		}
	}

	ShowImg(colorized_imgd);

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

	float width = 75, height = 50, depth = 45;

	for (int x = -width; x < width; ++x)
		for (int y = -height; y < height; ++y)
			for (int z = -depth; z < depth; ++z)
			{
				Construction3D.push_back(cv::Point3d(x, y, z));
			}

	std::cout << Construction3D.size() << std::endl;


	Rotate(params[1], params[2], params[0], Construction3D);

	//std::cout << "p.y.r.: " << params[0] << " " << params[1] << " " << params[2] << "\n";

	std::cout << "c.p.: " << central_3d_point.x << " " << central_3d_point.y << " " << central_3d_point.z << "\n";

	auto cp_dp_x = (int)(central_3d_point.x * fx_d) / central_3d_point.z + cx_d;
	auto cp_dp_y = (int)(central_3d_point.y * fy_d) / central_3d_point.z + cy_d;

	// imgd.at<ushort>(cp_dp_y, cp_dp_x) = 65000;

	for (auto i = 0; i < Construction3D.size(); ++i) {
		Construction3D[i] += central_3d_point;

		//std::cout << Construction3D[i].x << " " << Construction3D[i].y << " " << Construction3D[i].z << "\n";

		auto dp_x = (int)(Construction3D[i].x * fx_d) / Construction3D[i].z + cx_d;
		auto dp_y = (int)(Construction3D[i].y * fy_d) / Construction3D[i].z + cy_d;

		//std::cout << dp_x << " " << dp_y << " " << Construction3D[i].z << "\n";

		//P3D[0][0] = (i - cy_d) * depthValue / fy_d;
		//P3D[1][0] = (j - cx_d) * depthValue / fx_d;
		if (dp_x > 0 && dp_x < 512 && dp_y > 0 && dp_y < 424)
			if (imgd.at<ushort>(dp_y, dp_x) > Construction3D[i].z || imgd.at<ushort>(dp_y, dp_x) < 50)
				imgd.at<ushort>(dp_y, dp_x) = Construction3D[i].z;
	}

	// ShowImg(colorized_imgd);
	ColorizeDepth(imgd);

	//Post-processing cv Mat to byte array
	std::memcpy(imgResData, imgd.data, imgd.total() * imgd.elemSize() * sizeof(byte));
	
	success = true;

	return success;
}



//Rotates 3D points
EXPORT void Rotate(double pitch, double roll, double yaw, std::vector<cv::Point3d> &points) {
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

//Maps tha artificial HMD from depth to color (BGR image). Returns a byte array with the hmd-mapped Color data
EXPORT bool Depth2BGRA(const byte* imgColorData, const byte* imgDepthData, byte* imgResData)
{
	bool success = false;

	/* TBD */
	
	return success;
}

//Maps tha artificial HMD from depth to color (BGR image). Returns a byte array with the hmd-mapped Color data
EXPORT bool Depth2BGR(const byte* imgColorData, const byte* imgDepthData, byte* imgResData)
{
	bool success = false;

	/* TBD */

	return success;
}
